
import pickle
import networkx as nx
import random
import torch
from numpy.ma.core import append
from torch_geometric.data import Data
from sklearn.model_selection import KFold
import torch_geometric.utils as utils
import os



import torch.nn.functional as F

def read_graph_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            graph = pickle.load(file)
        return graph
    except FileNotFoundError:
        print(f"error: file {file_path} not found")
    except pickle.UnpicklingError:
        print("error: Unable to deserialize file contents")
    except Exception as e:
        print(f"An unknown error occurred: {e}")
    return None


def min_min_normalization(x):
    min_x = x.min(dim=0, keepdim=True).values
    max_x = x.max(dim=0, keepdim=True).values
    x = (x - min_x) / (max_x - min_x + 1e-8)

    return x


def robust_scale(tensor):
    """
    对输入的 PyTorch 张量进行稳健标准化
    :param tensor: 输入的 PyTorch 张量
    :return: 标准化后的 PyTorch 张量
    """
    # 计算中位数
    median = torch.median(tensor, dim=0)[0]
    # 计算第 25 百分位数
    q1 = torch.quantile(tensor, 0.25, dim=0)
    # 计算第 75 百分位数
    q3 = torch.quantile(tensor, 0.75, dim=0)
    # 计算四分位距
    iqr = q3 - q1
    # 避免除零错误，若四分位距为 0，则加上一个极小的常数
    iqr = torch.where(iqr == 0, torch.tensor(1e-8, dtype=iqr.dtype, device=iqr.device), iqr)
    # 进行稳健标准化
    scaled_tensor = (tensor - median) / iqr
    return scaled_tensor

def multidigraph_to_pyg(sub_G, sub_G_undirected):


    node_to_index = {node: index for index, node in enumerate(sub_G_undirected.nodes())}

    isp_count = 0

    node_labels = []
    for node, data in sub_G_undirected.nodes(data=True):
        node_label = data['isp']
        node_labels.append(node_label)
        if data['isp'] == 1:
            isp_count += 1

    y = torch.tensor(node_labels, dtype=torch.long)




    edge_index = []
    edge_features = []

    edge_count = 0
    for u, v, key, data in sub_G_undirected.edges(data=True, keys=True):
        u_index = node_to_index[u]
        v_index = node_to_index[v]

        edge_index.append([u_index, v_index])

        amount, timestamp = data['amount'], data['timestamp']
        edge_features.append([amount, timestamp])

        edge_count += 1

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    node_features = []
    for node in sub_G.nodes():
        # 入度
        in_degree = sub_G.in_degree(node)
        # 出度
        out_degree = sub_G.out_degree(node)
        # 总度
        total_degree = in_degree + out_degree
        # 发送到当前节点的 amount 总数
        in_amount_total = sum([data['amount'] for _, _, data in sub_G.in_edges(node, data=True)])
        # 当前节点发出的 amount 总数
        out_amount_total = sum([data['amount'] for _, _, data in sub_G.out_edges(node, data=True)])
        # 总交易数
        total_transaction_amount = in_amount_total + out_amount_total
        # 邻居数
        neighbors = set(sub_G.successors(node)).union(set(sub_G.predecessors(node)))
        neighbor_count = len(neighbors)
        # 第一笔交易和最后一笔交易的时间间隔
        in_timestamps = [data['timestamp'] for _, _, data in sub_G.in_edges(node, data=True)]
        out_timestamps = [data['timestamp'] for _, _, data in sub_G.out_edges(node, data=True)]
        all_timestamps = in_timestamps + out_timestamps
        if all_timestamps:
            time_interval = max(all_timestamps) - min(all_timestamps)
        else:
            time_interval = 0
        node_features.append([in_degree, out_degree, total_degree, in_amount_total, out_amount_total,
                           total_transaction_amount, neighbor_count, time_interval])

    # 将节点统计信息转换为 torch.Tensor 作为节点特征
    x = torch.tensor(node_features, dtype=torch.float)


    # Normalization
    x = min_min_normalization(x)
    edge_attr = min_min_normalization(edge_attr)

    x = robust_scale(x)
    edge_attr = robust_scale(edge_attr)


    #
    # x = F.normalize(x, p=2, dim=1)
    #
    # edge_attr = F.normalize(edge_attr, p=2, dim=1)
    # 创建 torch_geometric.Data 对象
    pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


    return pyg_graph, isp_count



def dataset_split(pyg_graph, seed=42):
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds = []
    for train_index, test_index in kf.split(pyg_graph.y):
        train_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool)
        train_mask[train_index] = True
        test_mask[test_index] = True

        # 复制 pyg_graph 并设置训练和测试掩码
        train_dataset = pyg_graph.clone()
        test_dataset = pyg_graph.clone()
        train_dataset.train_mask = train_mask
        test_dataset.test_mask = test_mask

        folds.append([train_dataset, test_dataset])

    return folds


def apply_data_augmentation(folds):
    augmented_folds = []
    for train_dataset, test_dataset in folds:
        train_dataset_aug_1 = []
        train_dataset_aug_2 = []

        # 应用数据增强
        drop_mask_1 = torch.empty(size=(train_dataset.x.size(0), train_dataset.x.size(1)), dtype=torch.float32).uniform_(0, 1) < 0.2
        drop_mask_2 = torch.empty(size=(train_dataset.x.size(0), train_dataset.x.size(1)), dtype=torch.float32).uniform_(0, 1) < 0.2

        train_dataset_aug_1 = train_dataset.clone()
        train_dataset_aug_2 = train_dataset.clone()

        # 对每一行应用不同的掩码
        for i in range(train_dataset.x.size(0)):
            train_dataset_aug_1.x[i, drop_mask_1[i]] = 0
            train_dataset_aug_2.x[i, drop_mask_2[i]] = 0


        augmented_folds.append([train_dataset_aug_1, train_dataset_aug_2, train_dataset, test_dataset])

    return augmented_folds

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes
    if key == 'subgraph_node_idx':
        return self.num_nodes
    if key == 'subgraph_indicator':
        return self.num_nodes
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0


def split_data_by_batch(data, batch_size):
    num_nodes = data.num_nodes
    data_list = []
    for i in range(0, num_nodes, batch_size):
        start = i
        end = min(i + batch_size, num_nodes)
        # 截取部分节点
        sub_data = data.clone()
        sub_data.x = data.x[start:end]
        sub_data.y = data.y[start:end]
        # 筛选出在当前子图中的边
        edge_mask = (data.edge_index[0] >= start) & (data.edge_index[0] < end) & (data.edge_index[1] >= start) & (data.edge_index[1] < end)
        sub_data.edge_index = data.edge_index[:, edge_mask] - start
        if hasattr(data, 'edge_attr'):
            sub_data.edge_attr = data.edge_attr[edge_mask]
        if hasattr(data, 'train_mask'):
            sub_data.train_mask = data.train_mask[start:end]
        if hasattr(data, 'test_mask'):
            sub_data.test_mask = data.test_mask[start:end]
        data_list.append(sub_data)
    return data_list


class GraphDataset(object):
    def __init__(self, dataset, degree=False, k_hop=2, se="gnn", use_subgraph_edge_attr=False,
                 cache_path=None, return_complete_index=False):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.degree = degree

        self.compute_degree()
        self.abs_pe_list = None
        self.return_complete_index = return_complete_index  # False
        self.k_hop = k_hop  # 2
        self.se = se  # gnn
        self.use_subgraph_edge_attr = use_subgraph_edge_attr  # True
        self.cache_path = cache_path
        if self.se == 'khopgnn':
            Data.__inc__ = my_inc
            self.extract_subgraphs()

    def compute_degree(self):

        # for data in self.dataset:
        #     assert data.edge_index.max() < data.num_nodes

        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx,
                    self.k_hop,
                    graph.edge_index,
                    relabel_nodes=True,
                    num_nodes=graph.num_nodes
                )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask])  # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data = self.dataset[index]

        # if self.n_features == 1:
        #     data.x = data.x.squeeze(-1)
        # if not isinstance(data.y, list):
        #     data.y = data.y.view(data.y.shape[0], -1)
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        if self.degree:
            data.degree = self.degree_list[index]
        # data.abs_pe = None
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset):
            data.abs_pe = self.abs_pe_list[index]

        # add subgraphs and relevant meta data
        if self.se == "khopgnn":
            if self.cache_path is not None:
                cache_file = torch.load("{}_{}.pt".format(self.cache_path, index))
                data.subgraph_edge_index = cache_file['subgraph_edge_index']
                data.num_subgraph_nodes = len(cache_file['subgraph_node_index'])
                data.subgraph_node_idx = cache_file['subgraph_node_index']
                data.subgraph_edge_attr = cache_file['subgraph_edge_attr']
                data.subgraph_indicator = cache_file['subgraph_indicator_index']
                return data
            data.subgraph_edge_index = self.subgraph_edge_index[index]
            data.num_subgraph_nodes = len(self.subgraph_node_index[index])
            data.subgraph_node_idx = self.subgraph_node_index[index]
            if self.use_subgraph_edge_attr and data.edge_attr is not None:
                data.subgraph_edge_attr = self.subgraph_edge_attr[index]
            data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)
        else:
            data.num_subgraph_nodes = None
            data.subgraph_node_idx = None
            data.subgraph_edge_index = None
            data.subgraph_indicator = None

        return data

if __name__ == '__main__':

    sub_G = read_graph_from_pickle('./datasets_phish_scam/directed-30.pkl')
    sub_G_undirected = read_graph_from_pickle('./datasets_phish_scam/undirected-30.pkl')


    pyg_graph, labeled_node = multidigraph_to_pyg(sub_G, sub_G_undirected)


    edge_count = sub_G_undirected.number_of_edges()

    node_count = sub_G_undirected.number_of_nodes()

    total_degree = sum(node_degree[1] for node_degree in list(sub_G_undirected.degree()))
    average_degree = total_degree / node_count

    print(f"Edge Count: {edge_count}")
    print(f"Average Degree: {average_degree}")
    print(f"Node Count: {node_count}")
    print(f"Labeled Node Count: {labeled_node}")

    split_data_set = dataset_split(pyg_graph)

    augmentation_data = apply_data_augmentation(split_data_set)

    batch_size = 4
    batch_data = []
    for train_dataset_aug_1, train_dataset_aug_2, train_dataset, test_dataset in augmentation_data:
        new_train_dataset_aug_1 = split_data_by_batch(train_dataset_aug_1, batch_size)
        new_train_dataset_aug_2 = split_data_by_batch(train_dataset_aug_2, batch_size)
        new_train_dataset = split_data_by_batch(train_dataset, batch_size)
        new_test_dataset = split_data_by_batch(test_dataset, batch_size)
        batch_data.append([new_train_dataset_aug_1, new_train_dataset_aug_2, new_train_dataset, new_test_dataset])