import pickle
import networkx as nx
import random
import torch
from numpy.ma.core import append
from torch_geometric.data import Data
from sklearn.model_selection import KFold
import torch_geometric.utils as utils
import os
import matplotlib.pyplot as plt
import re

import numpy as np

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


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data  # 假设使用PyG

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data  # 假设使用PyG


def visualize_graph_v2(pyg_graph):
    """可视化图数据，随机选择一个标签为1的节点并用星形符号显示"""
    # 创建NetworkX图
    G = nx.Graph()
    G.add_nodes_from(range(pyg_graph.num_nodes))
    edges = list(zip(pyg_graph.edge_index[0].tolist(), pyg_graph.edge_index[1].tolist()))
    G.add_edges_from(edges)

    # 节点颜色映射 (0=蓝色, 1=红色)
    node_colors = ['#E2ECEF' if label == 0 else '#C00000' for label in pyg_graph.y]

    # 计算节点度，用于加权布局和节点大小
    degrees = dict(G.degree())

    # 使用力导向布局算法，增加节点间距
    pos = nx.spring_layout(
        G,
        k=0.8 / np.sqrt(len(G.nodes())),  # 增加k值，使节点间距更大
        iterations=20, # 减少迭代次数，防止节点过于紧凑
        weight=None,  # 不考虑边权重
        seed=42  # 固定随机种子使结果可重现
    )

    # 自定义函数：调整节点位置使其分布在矩形区域
    def rectangular_layout_adjustment(pos, padding=0.6):
        """
        将节点位置调整为矩形分布，并增加松散度
        """
        # 获取当前布局的边界
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        max_y = max(p[1] for p in pos.values())

        width = max_x - min_x
        height = max_y - min_y

        # 计算宽高比，用于确定矩形形状
        aspect_ratio = width / height if height > 0 else 1.0

        # 调整节点位置，使其均匀分布在矩形内
        new_pos = {}
        for node, (x, y) in pos.items():
            # 将节点位置归一化到 [0,1] 范围
            norm_x = (x - min_x) / (width if width > 0 else 1)
            norm_y = (y - min_y) / (height if height > 0 else 1)

            # 添加更多随机性，增加松散度
            jitter_x = np.random.normal(0, 0.09)  # 增加抖动幅度
            jitter_y = np.random.normal(0, 0.09)  # 增加抖动幅度

            # 应用抖动并确保在 [padding, 1-padding] 范围内
            norm_x = max(padding, min(1 - padding, norm_x + jitter_x))
            norm_y = max(padding, min(1 - padding, norm_y + jitter_y))

            # 将归一化位置映射回画布大小
            new_pos[node] = (norm_x, norm_y)

        return new_pos

    # 应用矩形布局调整，增加padding使节点更松散
    pos = rectangular_layout_adjustment(pos, padding=0.2)

    # 可视化
    plt.figure(figsize=(12, 8))  # 增加画布大小

    # 绘制节点，根据度调整大小，但整体缩小一些
    node_sizes = [v * 8 + 12 for v in degrees.values()]  # 减小节点大小基数

    # 找出所有标签为1的节点
    label_1_nodes = [i for i, label in enumerate(pyg_graph.y) if label == 1]

    # 如果有标签为1的节点，选择度最高的一个
    star_node = None
    if label_1_nodes:
        # 找到度最高的标签为1的节点
        star_node = max(label_1_nodes, key=lambda x: degrees[x])
        # 从普通节点列表中移除被选中的星形节点
        normal_nodes = [i for i in G.nodes() if i != star_node]
    else:
        normal_nodes = list(G.nodes())

    # 绘制普通节点
    nx.draw_networkx_nodes(
        G, pos, nodelist=normal_nodes,
        node_size=[node_sizes[i] for i in normal_nodes],
        node_color=[node_colors[i] for i in normal_nodes],
        alpha=0.8, linewidths=0.5, edgecolors='#74AAAA'
    )

    # 如果有星形节点，绘制它
    if star_node is not None:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[star_node],
            node_size=node_sizes[star_node] * 1.8,  # 星形节点更大
            node_color=node_colors[star_node],
            alpha=0.9, linewidths=0.5, edgecolors='#74AAAA',
        )

    # 绘制边，根据节点距离调整透明度
    for edge in G.edges():
        n1, n2 = edge
        dist = np.linalg.norm(np.array(pos[n1]) - np.array(pos[n2]))
        alpha = max(0.2, min(0.6, 1.0 - dist * 1.5))  # 调整透明度范围
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge],
            width=0.6,  # 减小边的宽度
            alpha=alpha,
            edge_color='#999999'  # 浅灰色边
        )

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf(), G, star_node

def visualize_graph(pyg_graph):
    """可视化图数据，随机选择一个标签为1的节点并用星形符号显示"""
    # 创建NetworkX图
    G = nx.Graph()
    G.add_nodes_from(range(pyg_graph.num_nodes))
    edges = list(zip(pyg_graph.edge_index[0].tolist(), pyg_graph.edge_index[1].tolist()))
    G.add_edges_from(edges)

    # 节点颜色映射 (0=蓝色, 1=红色)
    node_colors = ['#E2ECEF' if label == 0 else '#C00000' for label in pyg_graph.y]

    # 计算节点度，用于加权布局
    degrees = dict(G.degree())

    # 使用改进的力导向布局算法
    pos = nx.spring_layout(
        G,
        k=0.4 / np.sqrt(len(G.nodes())),  # 根据节点数量动态调整间距
        iterations=150,  # 增加迭代次数使布局更稳定
        weight='weight',  # 考虑边权重
        seed=46  # 固定随机种子使结果可重现
    )

    # 为高连接节点增加额外排斥力，防止节点聚集
    for _ in range(20):  # 额外进行20次排斥力调整
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
                    if dist < 0.1:  # 如果距离过近
                        # 计算排斥方向
                        direction = (np.array(pos[node1]) - np.array(pos[node2])) / dist
                        # 轻微推开节点
                        pos[node1] += direction * 0.01
                        pos[node2] -= direction * 0.01

    # 可视化
    plt.figure(figsize=(10, 6))

    # 绘制节点，根据度调整大小
    node_sizes = [v * 10 + 15 for v in degrees.values()]

    # 找出所有标签为1的节点
    label_1_nodes = [i for i, label in enumerate(pyg_graph.y) if label == 1]

    # 如果有标签为1的节点，选择度最高的一个
    star_node = None
    if label_1_nodes:
        # 找到度最高的标签为1的节点
        star_node = max(label_1_nodes, key=lambda x: degrees[x])
        # 从普通节点列表中移除被选中的星形节点
        normal_nodes = [i for i in G.nodes() if i != star_node]
    else:
        normal_nodes = list(G.nodes())

    # 绘制普通节点
    nx.draw_networkx_nodes(
        G, pos, nodelist=normal_nodes,
        node_size=[node_sizes[i] for i in normal_nodes],
        node_color=[node_colors[i] for i in normal_nodes],
        alpha=0.9, linewidths=0.5, edgecolors='#74AAAA'
    )

    # 如果有星形节点，绘制它
    if star_node is not None:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[star_node],
            node_size=node_sizes[star_node] * 1.5,  # 星形节点更大
            node_color=node_colors[star_node],
            # node_shape='*',  # 使用星形符号
            alpha=0.9, linewidths=0.5, edgecolors='#74AAAA'
        )

    # 绘制边，根据节点距离调整透明度
    for edge in G.edges():
        n1, n2 = edge
        dist = np.linalg.norm(np.array(pos[n1]) - np.array(pos[n2]))
        alpha = max(0.3, min(0.7, 1.0 - dist * 2))  # 距离越远透明度越低
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge],
            width=0.8,
            alpha=alpha,
            edge_color='#999999'  # 浅灰色边
        )

    # 添加图例
    # plt.scatter([], [], c='#E2ECEF', s=50, label='0')
    # plt.scatter([], [], c='#C00000', s=50, label='1')
    # if star_node is not None:
    #     plt.scatter([], [], c='#C00000', s=75, marker='*', label='Selected 1')
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1)

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf(), G, star_node


def visualize_sub_graph_v2(nx_G, pyg_graph, star_node):
    """可视化图数据（仅一阶子图）"""
    # 提取star_node周围的一阶子图
    first_order = list(nx_G.neighbors(star_node))

    # 创建一阶子图
    subgraph_nodes = [star_node] + first_order
    G_sub = nx_G.subgraph(subgraph_nodes)

    # 为子图创建新的布局，增加节点间距
    pos_sub = nx.spring_layout(
        G_sub,
        k=0.6 / np.sqrt(len(G_sub.nodes())),  # 增加k值使节点更松散
        iterations=100,  # 减少迭代次数
        weight=None,  # 不考虑边权重
        seed=42
    )

    # 自定义函数：调整节点位置使其分布在矩形区域
    def rectangular_layout_adjustment(pos, padding=0.2):
        """将节点位置调整为矩形分布"""
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        max_y = max(p[1] for p in pos.values())

        width = max_x - min_x
        height = max_y - min_y

        new_pos = {}
        for node, (x, y) in pos.items():
            # 归一化节点位置
            norm_x = (x - min_x) / (width if width > 0 else 1)
            norm_y = (y - min_y) / (height if height > 0 else 1)

            # 添加随机抖动增加松散度
            jitter_x = np.random.normal(0, 0.00)
            jitter_y = np.random.normal(0, 0.00)

            # 应用抖动并确保在边界内
            norm_x = max(padding, min(1 - padding, norm_x + jitter_x))
            norm_y = max(padding, min(1 - padding, norm_y + jitter_y))

            new_pos[node] = (norm_x, norm_y)

        return new_pos

    # 应用矩形布局调整
    pos_sub = rectangular_layout_adjustment(pos_sub)

    # 节点颜色映射 (0=蓝色, 1=红色)
    node_colors = ['#E2ECEF' if label == 0 else '#C00000' for label in pyg_graph.y]

    # 子图节点颜色
    sub_node_colors = [node_colors[i] for i in subgraph_nodes]

    # 子图节点大小
    sub_degrees = dict(G_sub.degree())
    sub_node_sizes = [sub_degrees[i] * 12 + 18 for i in subgraph_nodes]  # 减小节点大小

    # 区分不同阶的节点
    star_node_idx = subgraph_nodes.index(star_node)
    first_order_indices = [subgraph_nodes.index(i) for i in first_order]

    # 可视化一阶子图，增大画布
    plt.figure(figsize=(5, 3))

    # 绘制一阶节点
    nx.draw_networkx_nodes(
        G_sub, pos_sub,
        nodelist=[subgraph_nodes[i] for i in first_order_indices],
        node_size=[sub_node_sizes[i] for i in first_order_indices],
        node_color=[sub_node_colors[i] for i in first_order_indices],
        alpha=0.8, linewidths=0.5, edgecolors='#74AAAA'
    )

    # 绘制中心节点
    nx.draw_networkx_nodes(
        G_sub, pos_sub,
        nodelist=[subgraph_nodes[star_node_idx]],
        node_size=sub_node_sizes[star_node_idx] * 1.5,
        node_color=sub_node_colors[star_node_idx],

        alpha=1.0, linewidths=1.0, edgecolors='#C00000'
    )

    # 检测子图中的环
    cycles = list(nx.simple_cycles(G_sub))

    # 构建环边集合
    cycle_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            cycle_edges.add(tuple(sorted([u, v])))

    # 绘制子图的边，区分环边和非环边
    for edge in G_sub.edges():
        n1, n2 = edge
        sorted_edge = tuple(sorted([n1, n2]))

        # 判断是否为环边
        if sorted_edge in cycle_edges:
            edge_color = '#2E54A1'  # 蓝色表示环边
            width = 1.3
            alpha = 0.8
        else:
            edge_color = '#666666'  # 普通边为灰色
            width = 1.0
            alpha = max(0.2, min(0.6, 1.0 - np.linalg.norm(np.array(pos_sub[n1]) - np.array(pos_sub[n2])) * 1.5))

        nx.draw_networkx_edges(
            G_sub, pos_sub,
            edgelist=[edge],
            width=width,
            alpha=alpha,
            edge_color=edge_color
        )

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf(), G_sub, star_node


def visualize_sub_graph(nx_G, pyg_graph, star_node):
    """可视化图数据（仅一阶子图）"""
    # 提取star_node周围的一阶子图
    first_order = list(nx_G.neighbors(star_node))

    # 创建一阶子图
    subgraph_nodes = [star_node] + first_order
    G_sub = nx_G.subgraph(subgraph_nodes)

    # 为子图创建新的布局
    pos_sub = nx.spring_layout(
        G_sub,
        k=0.4 / np.sqrt(len(G_sub.nodes())),
        iterations=150,
        weight='weight',
        seed=42
    )

    # 节点颜色映射 (0=蓝色, 1=红色)
    node_colors = ['#E2ECEF' if label == 0 else '#C00000' for label in pyg_graph.y]

    # 子图节点颜色
    sub_node_colors = [node_colors[i] for i in subgraph_nodes]

    # 子图节点大小
    sub_degrees = dict(G_sub.degree())
    sub_node_sizes = [sub_degrees[i] * 15 + 20 for i in subgraph_nodes]

    # 区分不同阶的节点
    star_node_idx = subgraph_nodes.index(star_node)
    first_order_indices = [subgraph_nodes.index(i) for i in first_order]

    # 可视化一阶子图
    plt.figure(figsize=(3, 3))

    # 绘制一阶节点
    nx.draw_networkx_nodes(
        G_sub, pos_sub,
        nodelist=[subgraph_nodes[i] for i in first_order_indices],
        node_size=[sub_node_sizes[i] for i in first_order_indices],
        node_color=[sub_node_colors[i] for i in first_order_indices],
        alpha=0.9, linewidths=0.5, edgecolors='#74AAAA'
    )

    # 绘制中心节点
    nx.draw_networkx_nodes(
        G_sub, pos_sub,
        nodelist=[subgraph_nodes[star_node_idx]],
        node_size=sub_node_sizes[star_node_idx] * 1.5,
        node_color=sub_node_colors[star_node_idx],
        # node_shape='*',
        alpha=1.0, linewidths=1.0, edgecolors='#C00000'
    )

    # 检测子图中的环
    cycles = list(nx.simple_cycles(G_sub))

    # 构建环边集合（无向边，以元组形式存储，且保证元组内节点有序）
    cycle_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            # 确保边元组按节点顺序排列，避免重复
            cycle_edges.add(tuple(sorted([u, v])))

    # 绘制子图的边，区分环边和非环边
    for edge in G_sub.edges():
        n1, n2 = edge
        sorted_edge = tuple(sorted([n1, n2]))

        # 判断是否为环边
        if sorted_edge in cycle_edges:
            edge_color = '#2E54A1'  # 蓝色表示环边
            width = 1.5
            alpha = 0.8
        else:
            edge_color = '#333333'  # 普通边为深色
            width = 1.2
            alpha = max(0.3, min(0.7, 1.0 - np.linalg.norm(np.array(pos_sub[n1]) - np.array(pos_sub[n2])) * 2))

        nx.draw_networkx_edges(
            G_sub, pos_sub,
            edgelist=[edge],
            width=width,
            alpha=alpha,
            edge_color=edge_color
        )


    plt.axis('off')
    plt.tight_layout()

    return plt.gcf(), G_sub, star_node



if __name__ == '__main__':


    data = 'directed-1000.pkl'

    sub_G = read_graph_from_pickle(os.path.join('./datasets_phish_scam', data))
    sub_G_undirected = read_graph_from_pickle(os.path.join('./datasets_phish_scam', 'un'+data))

    pyg_graph, labeled_node = multidigraph_to_pyg(sub_G, sub_G_undirected)

    edge_count = sub_G_undirected.number_of_edges()

    node_count = sub_G_undirected.number_of_nodes()

    total_degree = sum(node_degree[1] for node_degree in list(sub_G_undirected.degree()))
    average_degree = total_degree / node_count

    print(f"Edge Count: {edge_count}")
    print(f"Average Degree: {average_degree}")
    print(f"Node Count: {node_count}")
    print(f"Labeled Node Count: {labeled_node}")

    # 可视化图
    fig, nx_G, star_node = visualize_graph(pyg_graph)

    # 保存图像
    fig.savefig(os.path.join('./datasets_phish_scam', re.findall("\d+", data)[0] + '_visual.png'), dpi=300, bbox_inches='tight')
    print("save as 'visual.png'")


    sub_fig, G_sub, star_node = visualize_sub_graph(nx_G, pyg_graph, star_node)

    # 保存图像
    sub_fig.savefig(os.path.join('./datasets_phish_scam', re.findall("\d+", data)[0] + '_sub_visual.png'), dpi=300, bbox_inches='tight')
    print("save as 'sub_visual.png'")