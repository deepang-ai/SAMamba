import os
import os.path as osp
import glob
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

names = ['A',
         'graph_indicator', 'graph_labels', 'graph_attributes',
         'node_labels', 'node_attributes',
         'edge_labels', 'edge_attributes',
         'node_importance_labels'
         ]


def my_read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1

    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = important_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')

    # if 'node_labels' in names:
    #     node_labels = read_file(folder, prefix, 'node_labels', torch.long)
    #     if node_labels.dim() == 1:
    #         node_labels = node_labels.unsqueeze(-1)
    #     node_labels = node_labels - node_labels.min(dim=0)[0]
    #     node_labels = node_labels.unbind(dim=-1)
    #     node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
    #     node_labels = torch.cat(node_labels, dim=-1).to(torch.float)

    if 'node_importance_labels' in names:
        important_labels = read_file(folder, prefix, 'node_importance_labels', torch.long)
        if important_labels.dim() == 1:
            important_labels = important_labels.unsqueeze(-1)
        important_labels = important_labels - important_labels.min(dim=0)[0]
        important_labels = important_labels.unbind(dim=-1)
        important_labels = [F.one_hot(x, num_classes=-1) for x in important_labels]
        important_labels = torch.cat(important_labels, dim=-1).to(torch.float)
        # np.savetxt('./test/important_labels_one_hot.csv', important_labels.detach().numpy(), fmt='%d', delimiter=',')
    # num_node_attributes = node_attributes.size()[1]
    # num_node_labels = node_labels.size()[1]
    # num_node_dd_labels = important_labels.size()[1]

    x = cat([node_attributes, important_labels])



    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    # if 'edge_labels' in names:
    #     edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
    #     if edge_labels.dim() == 1:
    #         edge_labels = edge_labels.unsqueeze(-1)
    #     edge_labels = edge_labels - edge_labels.min(dim=0)[0]
    #     edge_labels = edge_labels.unbind(dim=-1)
    #     edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
    #     edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    # edge_attr = cat([edge_attributes, edge_labels])
    edge_attr = edge_attributes.to(torch.float32)

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    return data, slices


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


from torch_geometric.data import InMemoryDataset
from typing import Optional, Callable, List

class MyBlockChain_TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False, use_node_importance=False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        ####################################
        # print(self.data, self.slices)
        #Data(x=[6274, 14888], edge_index=[2, 10875], edge_attr=[10875, 2], y=[146])
        ####################################
        if self.data.x is not None:
            if not use_node_attr and not use_node_importance:         # replace with identity matrix
                self.data.x = torch.eye(self.data.x.size(0))
            elif not use_node_importance:
                num_node_importance = self.num_node_importance
                self.data.x = self.data.x[:,:-num_node_importance]    # drop node importance
            elif not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]    # drop node attribute



        if self.data.edge_attr is not None and not use_edge_attr:
            # num_edge_attributes = self.num_edge_attributes
            # self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            self.data.edge_attr = torch.ones_like(self.data.edge_attr)     # replace with 1s matrix





    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, name)

    @property
    def num_node_labels(self) -> int:
        # if self.data.x is None:
        #     return 0
        # for i in range(self.data.x.size(1)):
        #     x = self.data.x[:, i:]
        #     if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
        #         return self.data.x.size(1) - i
        return 0

    @property
    def num_node_importance(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, -i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels - self.num_node_importance

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator', 'graph_labels', 'node_attributes', 'edge_attributes', 'node_importance_labels']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # url = self.cleaned_url if self.cleaned else self.url
        # folder = osp.join(self.root, self.name)
        # path = download_url(f'{url}/{self.name}.zip', folder)
        # extract_zip(path, folder)
        # os.unlink(path)
        # shutil.rmtree(self.raw_dir)
        # os.rename(osp.join(folder, self.name), self.raw_dir)
        return


    #https://blog.csdn.net/misite_J/article/details/118282680?ops_request_misc=&request_id=&biz_id=102&utm_term=InMemoryDataset.collate()&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-118282680.142^v96^pc_search_result_base1&spm=1018.2226.3001.4187
    def process(self):
        self.data, self.slices = my_read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'



if __name__ == '__main__':
    # folder = './data/eth/ico_wallets/2hop-20/averVolume/raw/'
    # prefix = 'ETHG'
    #
    # data, slices = my_read_tu_data(folder, prefix)
    # print(data)
    # print(slices)

    # dataset = MyBlockChain_TUDataset(root='./data/eth/ico_wallets/2hop-20/averVolume/', name='ETHG',
    #                                  use_node_attr=True,
    #                                  use_node_importance=False,
    #                                  use_edge_attr=False,
    #                                  transform=None)
    #
    # num_edge_list = []
    # num_node_list = []
    # for i in range(len(dataset)):
    #     num_edge_list.append(len(dataset[i].edge_index[0]))
    #     num_node_list.append(len(dataset[i].x))
    #
    # edge = pd.read_csv('data/eth/ico_wallets/2hop-20/averVolume/raw/ETHG_A.txt', header=None).values - 1
    #
    # #原邻接矩阵在所有节点上构建，此处转为在每一个子图上构建
    # num_edge_flag = 0
    # num_node_flag = 0
    # for index, (num_node, num_edge) in enumerate(zip(num_node_list, num_edge_list)):
    #     if index == 0:
    #         pass
    #     else:
    #         edge[num_edge_flag:num_edge_flag+num_edge,:] -= num_node_flag
    #
    #     num_node_flag += num_node
    #     num_edge_flag += num_edge
    #
    #
    # edge_feat = pd.read_csv('data/eth/ico_wallets/2hop-20/averVolume/raw/ETHG_edge_attributes.txt', header=None)
    # node_feat = pd.read_csv('data/eth/ico_wallets/2hop-20/averVolume/raw/ETHG_node_attributes.txt', header=None)
    # graph_label = pd.read_csv('data/eth/ico_wallets/2hop-20/averVolume/raw/ETHG_graph_labels.txt', header=None)
    #
    # np.savetxt('ico_wallets/averVolume/raw/edge.csv', edge, fmt='%d', delimiter=',')
    # np.savetxt('ico_wallets/averVolume/raw/edge-feat.csv', edge_feat, fmt='%.16f', delimiter=',')
    # np.savetxt('ico_wallets/averVolume/raw/node-feat.csv', node_feat, fmt='%d', delimiter=',')
    # np.savetxt('ico_wallets/averVolume/raw/graph-label.csv', graph_label, fmt='%d')
    # np.savetxt('ico_wallets/averVolume/raw/num-edge-list.csv', num_edge_list, fmt='%d')
    # np.savetxt('ico_wallets/averVolume/raw/num-node-list.csv', num_node_list, fmt='%d')

    ########################################
    #
    load_datasets = ['./data/eth/ico_wallets/2hop-20/averVolume/', './data/eth/ico_wallets/2hop-20/Times/',
                     './data/eth/ico_wallets/2hop-20/Volume/',
                     './data/eth/mining/2hop-20/averVolume/', './data/eth/mining/2hop-20/Times/',
                     './data/eth/mining/2hop-20/Volume/',
                     './data/eth/exchange/2hop-20/averVolume/', './data/eth/exchange/2hop-20/Times/',
                     './data/eth/exchange/2hop-20/Volume/',
                     './data/eth/phish_hack/2hop-20/averVolume/', './data/eth/phish_hack/2hop-20/Times/',
                     './data/eth/phish_hack/2hop-20/Volume/',]
    save_datasets = ['./ico_wallets/averVolume/', './ico_wallets/Times/', './ico_wallets/Volume/',
                     './mining/averVolume/', './mining/Times/', './mining/Volume/',
                     './exchange/averVolume/', './exchange/Times/', './exchange/Volume/',
                     # './phish_hack/averVolume/', './phish_hack/Times/', './phish_hack/Volume/'
                     ]

    # load_datasets = ['./data/eth/phish_hack/2hop-20/Volume/']
    # save_datasets = ['./phish_hack/Volume/']
    for load_dir, save_dir in zip(load_datasets, save_datasets):
        dataset = MyBlockChain_TUDataset(root=load_dir, name='ETHG',
                                         use_node_attr=True,
                                         use_node_importance=False,
                                         use_edge_attr=False,
                                         transform=None)

        num_edge_list = []
        num_node_list = []
        for i in range(len(dataset)):
            num_edge_list.append(len(dataset[i].edge_index[0]))
            num_node_list.append(len(dataset[i].x))

        edge = pd.read_csv(load_dir+'/raw/ETHG_A.txt', header=None).values - 1

        #原邻接矩阵在所有节点上构建，此处转为在每一个子图上构建
        num_edge_flag = 0
        num_node_flag = 0
        for index, (num_node, num_edge) in enumerate(zip(num_node_list, num_edge_list)):
            if index == 0:
                pass
            else:
                edge[num_edge_flag:num_edge_flag+num_edge,:] -= num_node_flag

            num_node_flag += num_node
            num_edge_flag += num_edge


        edge_feat = pd.read_csv(load_dir+'raw/ETHG_edge_attributes.txt', header=None)
        node_feat = pd.read_csv(load_dir+'raw/ETHG_node_attributes.txt', header=None)
        graph_label = pd.read_csv(load_dir+'raw/ETHG_graph_labels.txt', header=None)

        np.savetxt(save_dir+'raw/edge.csv', edge, fmt='%d', delimiter=',')
        np.savetxt(save_dir+'raw/edge-feat.csv', edge_feat, fmt='%.16f', delimiter=',')
        np.savetxt(save_dir+'raw/node-feat.csv', node_feat, fmt='%d', delimiter=',')
        np.savetxt(save_dir+'raw/graph-label.csv', graph_label, fmt='%d')
        np.savetxt(save_dir+'raw/num-edge-list.csv', num_edge_list, fmt='%d')
        np.savetxt(save_dir+'raw/num-node-list.csv', num_node_list, fmt='%d')

    ########################################
    # dataset = MyBlockChain_TUDataset(root='./data/eth/phish_hack/2hop-20/averVolume/', name='ETHG',
    #                                  use_node_attr=True,
    #                                  use_node_importance=False,
    #                                  use_edge_attr=False,
    #                                  transform=None)
    #
    # num_edge_list = []
    # num_node_list = []
    # for i in range(len(dataset)):
    #     num_edge_list.append(len(dataset[i].edge_index[0]))
    #     num_node_list.append(len(dataset[i].x))
    #
    # edge = pd.read_csv('data/eth/phish_hack/2hop-20/averVolume/raw/ETHG_A.txt', header=None).values - 1
    #
    # #原邻接矩阵在所有节点上构建，此处转为在每一个子图上构建
    # num_edge_flag = 0
    # num_node_flag = 0
    # for index, (num_node, num_edge) in enumerate(zip(num_node_list, num_edge_list)):
    #     if index == 0:
    #         pass
    #     else:
    #         edge[num_edge_flag:num_edge_flag+num_edge,:] -= num_node_flag
    #
    #     num_node_flag += num_node
    #     num_edge_flag += num_edge
    #
    #
    # edge_feat = pd.read_csv('data/eth/phish_hack/2hop-20/averVolume/raw/ETHG_edge_attributes.txt', header=None)
    # node_feat = pd.read_csv('data/eth/phish_hack/2hop-20/averVolume/raw/ETHG_node_attributes.txt', header=None)
    # graph_label = pd.read_csv('data/eth/phish_hack/2hop-20/averVolume/raw/ETHG_graph_labels.txt', header=None)
    #
    # np.savetxt('phish_hack/averVolume/raw/edge.csv', edge, fmt='%d', delimiter=',')
    # np.savetxt('phish_hack/averVolume/raw/edge-feat.csv', edge_feat, fmt='%.16f', delimiter=',')
    # np.savetxt('phish_hack/averVolume/raw/node-feat.csv', node_feat, fmt='%d', delimiter=',')
    # np.savetxt('phish_hack/averVolume/raw/graph-label.csv', graph_label, fmt='%d')
    # np.savetxt('phish_hack/averVolume/raw/num-edge-list.csv', num_edge_list, fmt='%d')
    # np.savetxt('phish_hack/averVolume/raw/num-node-list.csv', num_node_list, fmt='%d')