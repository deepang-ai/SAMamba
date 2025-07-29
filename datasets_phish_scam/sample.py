import pickle
import random

import networkx as nx
import argparse

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

def graph_sample(G, G_undirected, sample_size):

    isp_1_nodes = [nd for nd in G.nodes if G.nodes[nd]['isp'] == 1]



    if not isp_1_nodes:
        print("There are no nodes with 'isp' feature equal to 1, so sampling cannot be performed.")
        return None

    # start_node = random.choice([nd for nd in G.nodes])
    start_node = random.choice(isp_1_nodes)

    sampled_nodes = {start_node}


    current_node = start_node
    while len(sampled_nodes) < sample_size:
        neighbors = list(G_undirected.neighbors(current_node))
        if neighbors:

            weights = []
            for neighbor in neighbors:
                if G_undirected.nodes[neighbor].get('isp') == 1:

                    weights.append(9.0)
                else:
                    weights.append(1.0)


            next_node = random.choices(neighbors, weights=weights)[0]
            sampled_nodes.add(next_node)
            current_node = next_node
        else:

            start_node = random.choice(isp_1_nodes)
            current_node = start_node
            sampled_nodes.add(start_node)
    sub_G = G.subgraph(sampled_nodes)
    sub_G_undirected = G_undirected.subgraph(sampled_nodes)


    return sub_G, sub_G_undirected


def main():
    parser = argparse.ArgumentParser(description='Graph sampling script with specified sample size')
    parser.add_argument('--sample-size', type=int, required=True,
                        help='Number of nodes to sample from the graph')
    parser.add_argument('--input-file', type=str,
                        default='./datasets_phish_scam/MulDiGraph.pkl',
                        help='Path to the input pickle file containing the graph')
    parser.add_argument('--output-dir', type=str,
                        default='./datasets_phish_scam/',
                        help='Directory to save the output sampled graphs')
    args = parser.parse_args()

    # 读取图数据
    print("Reading graph from pickle file...")
    G = read_graph_from_pickle(args.input_file)
    if G is None:
        print("Failed to read graph. Exiting.")
        return

    # 转换为无向图
    print("Transforming to undirected graph...")
    G_undirected = G.to_undirected()
    print("Transformation done.")
    print(f"Original graph: {G}")

    # 进行图采样
    print(f"Sampling {args.sample_size} nodes from the graph...")
    sub_G, sub_G_undirected = graph_sample(G, G_undirected, args.sample_size)

    if sub_G is None or sub_G_undirected is None:
        print("Sampling failed. Exiting.")
        return

    # 计算并打印子图统计信息
    edge_count = sub_G_undirected.number_of_edges()
    node_count = sub_G_undirected.number_of_nodes()
    total_degree = sum(degree for _, degree in sub_G_undirected.degree())
    average_degree = total_degree / node_count if node_count > 0 else 0

    isp_count = 0
    for nd in sub_G_undirected:
        if G.nodes[nd]['isp'] == 1:
            isp_count += 1

    print(f"Sampling results:")
    print(f"Edge Count: {edge_count}")
    print(f"Average Degree: {average_degree:.2f}")
    print(f"Node Count: {node_count}")
    print(f"Labeled Node Count (isp=1): {isp_count}")

    # 保存采样结果
    directed_path = f"{args.output_dir}directed-{args.sample_size}.pkl"
    undirected_path = f"{args.output_dir}undirected-{args.sample_size}.pkl"

    with open(directed_path, "wb") as f:
        pickle.dump(nx.MultiDiGraph(sub_G), f)

    with open(undirected_path, "wb") as f:
        pickle.dump(nx.MultiGraph(sub_G_undirected), f)

    print(f"Sampled graphs saved to:")
    print(f"Directed: {directed_path}")
    print(f"Undirected: {undirected_path}")
    print("Done!")


if __name__ == "__main__":
    main()