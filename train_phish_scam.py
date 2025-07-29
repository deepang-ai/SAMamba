
import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import torch
from samamba.utils import count_parameters
from datetime import datetime
from timeit import default_timer as timer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = 'cpu'
from datasets_phish_scam.phish_scam_dataset_pyg import read_graph_from_pickle, multidigraph_to_pyg, apply_data_augmentation, dataset_split, GraphDataset, split_data_by_batch
import torch_geometric.utils as utils
from samamba.models import GraphEncoder

from torch.cuda import memory_allocated, max_memory_allocated, reset_peak_memory_stats
from thop import profile
from torch_geometric.data import Data,DataLoader
from samamba.position_encoding import POSENCODINGS

GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
    'rwgnn', 'khopgnn', 'gat'
]

from samamba.utils import Logger

def load_args():
    parser = argparse.ArgumentParser(
        description='SAMamba',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--seed', type=int, default=0,
    #                     help='random seed')

    parser.add_argument('--model', type=str, default='SAMamba',
                        help='model choices')
    parser.add_argument('--directed-graph-path', type=str, default='./datasets_phish_scam/directed-30000.pkl',
                        help='directed graph path')
    parser.add_argument('--undirected-graph-path', type=str, default='./datasets_phish_scam/undirected-30000.pkl',
                        help='undirected graph path')

    parser.add_argument('--dropout', type=float, default=0.1, help="dropout") #0.6
    parser.add_argument('--dim-hidden', type=int, default=64, help="hidden dimension")

    parser.add_argument('--num-heads', type=int, default=1, help="number of heads")


    parser.add_argument('--k-hop', type=int, default=1, help="number of hops")

    parser.add_argument('--se', type=str, default="gnn",  # k-subgraph or k-subtree GNN extractor
                        help='Extractor type: khopgnn, or gnn')
    parser.add_argument('--ie', type=str, default="mamba",
                        help='information encoder type: transformer, or mamba')

    parser.add_argument('--use-edge-attr', type=bool, default=True, help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=64, help='edge features hidden dim')



    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--num-layers', type=int, default=2, help="number of layers")

    parser.add_argument('--batch-size', type=int, default=4, help="batch size")
    parser.add_argument('--gnn-type', type=str, default='gat',  # Base GNN model
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')


    parser.add_argument('--pe',  type=bool, default=False, help='use position encoding')
    parser.add_argument('--abs-pe', type=str, default='lap', choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')

    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add', 'gat'],
                        # Aggregate node-level representations into a graph representation.
                        help='global pooling method')

    parser.add_argument('--profile', action='store_true', help='FLOPs and GPU memory statistics')

    parser.add_argument('--order', type=str, default="sd", choices=['nd', 'sd'],
                        help='node prioritization strategy in graph mamba: node degree, or subgraph degree')
    parser.add_argument('--task', type=str, default="node", choices=['graph', 'node'],
                        help='classification task')

    parser.add_argument('--training-times', '-t-times', type=int, help='training times', default=3)




    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()



    return args



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0

    tic = timer()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, _, out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].long())
        _, pred = out[data.train_mask].max(dim=1)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= len(loader.dataset)
    toc = timer()

    time = toc - tic

    return train_loss, time



def eval_epoch(model, loader, criterion, device):
    model.eval()
    ys, preds = [], []
    val_loss = 0

    tic = timer()
    for data in loader:
        data = data.to(device)
        _, _, out = model(data)
        loss = criterion(out[data.test_mask], data.y[data.test_mask].long())
        val_loss += loss.item() * data.num_graphs
        _, pred = out[data.test_mask].max(dim=1)
        ys.append(data.y[data.test_mask].cpu())
        preds.append(pred.cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    val_loss /= len(loader.dataset)

    toc = timer()

    time = toc - tic

    f1 = f1_score(y, pred, average=None)
    mf1 = f1_score(y, pred, average='micro')
    precision = precision_score(y, pred, average=None)
    recall = recall_score(y, pred, average=None)

    return val_loss, f1, mf1, precision, recall, time


def get_flops(model, input_data):
    if not isinstance(model, torch.nn.Module):
        print("Warning: current model do not support FLOPs statistic (Non Pytorch model)")
        return 0, 0
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(input_data,), verbose=False)
    return flops

def get_gpu_memory():

    if not torch.cuda.is_available():
        return {
            'allocated': 0,
            'max_allocated': 0
        }
    reset_peak_memory_stats()  # 重置峰值统计
    torch.cuda.synchronize()   # 确保统计同步
    allocated = memory_allocated() / 1024**2  # 已分配内存
    max_allocated = max_memory_allocated() / 1024**2  # 峰值内存
    return {
        'allocated': allocated,
        'max_allocated': max_allocated
    }


def main():
    args = load_args()
    args.batch_norm = not args.layer_norm

    if args.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'


    logging_dir = os.getcwd() + '/outdir/' + args.undirected_graph_path + '-' + args.model + '-' + str(
        datetime.now()).replace(' ', '-')

    Logger(logging_dir)

    print("Loading Data...")
    print(args.undirected_graph_path)

    sub_G = read_graph_from_pickle(args.directed_graph_path)
    sub_G_undirected = read_graph_from_pickle(args.undirected_graph_path)


    edge_count = sub_G_undirected.number_of_edges()

    node_count = sub_G_undirected.number_of_nodes()

    total_degree = sum(node_degree[1] for node_degree in list(sub_G_undirected.degree()))
    average_degree = total_degree / node_count

    print(f"Graph Edge Count: {edge_count}")
    print(f"Graph Average Degree: {average_degree}")
    print(f"Graph Node Count: {node_count}")


    un_edge_count = sub_G_undirected.number_of_edges()

    un_node_count = sub_G_undirected.number_of_nodes()

    un_total_degree = sum(node_degree[1] for node_degree in list(sub_G_undirected.degree()))
    un_average_degree = un_total_degree / un_node_count


    pyg_graph, labeled_node = multidigraph_to_pyg(sub_G, sub_G_undirected)

    print(f"Undirected Graph Edge Count: {un_edge_count}")
    print(f"Undirected Graph Average Degree: {un_average_degree}")
    print(f"Undirected Graph Node Count: {un_node_count}")
    print(f"Labeled Node Count: {labeled_node}")

    split_data_set = dataset_split(pyg_graph)



    # 自动计算weight
    all_labels = []
    for k, (train_dataset, test_dataset) in enumerate(split_data_set):
        all_labels.extend(train_dataset.y.cpu().numpy())
        all_labels.extend(test_dataset.y.cpu().numpy())
    unique, counts = np.unique(all_labels, return_counts=True)
    weight = 1. / torch.tensor(counts, dtype=torch.float)
    weight = weight / weight.sum() * len(unique)
    weight = weight.to(device)



    t_times_test_precision_list = []
    t_times_test_recall_list = []
    t_times_test_micro_f1_score_list = []
    t_times_test_f1_score_list = []

    t_times_infer_elapsed_list = []


    t_times_total_time_list = []
    t_times_epoch_time_list = []
    t_times_total_epoch_list = []

    for t in range(args.training_times):
        print("========================training times:{}========================".format(t))

        test_precision_list = []
        test_recall_list = []
        test_micro_f1_score_list = []
        test_f1_score_list = []

        infer_elapsed_list = []


        total_time_list = []
        epoch_time_list = []
        total_epoch_list = []


        for k, ( train_dataset, test_dataset) in enumerate(split_data_set):

            train_dset = GraphDataset([train_dataset], degree=True,
                                      k_hop=args.k_hop, se=args.se)

            test_dset= GraphDataset([test_dataset], degree=True,
                                      k_hop=args.k_hop, se=args.se)


            train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)


            if args.pe:
                print("Apply Position Encoding...")
                abs_pe_encoder = None
                if args.abs_pe and args.abs_pe_dim > 0:
                    abs_pe_method = POSENCODINGS[args.abs_pe]
                    abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
                    if abs_pe_encoder is not None:
                        abs_pe_encoder.apply_to(train_dset)
                        abs_pe_encoder.apply_to(test_dset)

            input_size = train_dataset.x.shape[1]
            num_classes = 2
            num_edge_features = train_dataset.edge_attr.shape[1]

            if 'pna' in args.gnn_type or args.gnn_type == 'mpnn':
                deg = torch.cat([
                    utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dset])
            else:
                deg = None


            model = GraphEncoder(in_size=input_size,
                                     num_class=num_classes,
                                     d_model=args.dim_hidden,
                                     dim_feedforward=int(2 * args.dim_hidden),
                                     dropout=args.dropout,
                                     num_heads=args.num_heads,
                                     num_layers=args.num_layers,
                                     batch_norm=args.batch_norm,
                                     abs_pe=args.abs_pe,
                                     abs_pe_dim=args.abs_pe_dim,
                                     gnn_type=args.gnn_type,
                                     k_hop=args.k_hop,
                                     use_edge_attr=args.use_edge_attr,
                                     num_edge_features=num_edge_features,
                                     edge_dim=args.edge_dim,
                                     se=args.se,
                                     ie=args.ie,
                                     deg=deg,
                                     in_embed=False,
                                     edge_embed=False,
                                     global_pool=args.global_pool,
                                     order=args.order,
                                     task=args.task
                                     )


            model.to(device)

            if args.profile:

                loader = DataLoader(train_dset, batch_size=1, shuffle=False)

                dummy_data = next(iter(loader)).to(device) if args.use_cuda else next(iter(loader))

                flops = get_flops(model, dummy_data)
                print(f"Model {args.model} Statistic：")
                print(f"FLOPs: {flops / 1e9:.8f} G")

                print("Total number of parameters: {} M".format(count_parameters(model) / 1e6))

            print(args.model)

            lr = args.lr
            epoches = args.epochs

            weight = torch.tensor([0.3, 0.7]).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss(weight)

            best_epoch = 0
            best_precision = 0
            best_recall = 0
            best_f1 = 0
            best_micro_f1 = 0

            print("Start Training...")
            total_start_time = timer()
            total_epoch = 0


            # ----------------------
            # GPU Memory before training
            # ----------------------
            if args.use_cuda and args.profile:
                print(f"GPU device: {torch.cuda.get_device_name(0)}")
                mem_before = get_gpu_memory()
                print(
                    f"GPU Memory before training: allocated {mem_before['allocated']:.2f} MB, Max {mem_before['max_allocated']:.2f} MB")

            for epoch in range(epoches):


                if args.use_cuda and args.profile:
                    reset_peak_memory_stats()
                train_loss, train_time = train_epoch(model, train_loader, criterion, optimizer, device)
                # 训练后获取内存使用
                if args.use_cuda and args.profile:
                    mem_train = get_gpu_memory()
                    print(
                        f"Epoch {epoch + 1} GPU Memory after training: allocated {mem_train['allocated']:.2f} MB, Max {mem_train['max_allocated']:.2f} MB")


                if (epoch + 1) % 1 == 0:

                    import time
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    if args.use_cuda and args.profile:
                        reset_peak_memory_stats()
                    val_loss, f1, mf1, precision, recall, test_time = eval_epoch(model, test_loader, criterion, device)
                    if args.use_cuda and args.profile:
                        mem_infer = get_gpu_memory()
                        print(
                            f"Epoch {epoch + 1} GPU Memory after training: allocated {mem_infer['allocated']:.2f} MB, Max {mem_infer['max_allocated']:.2f} MB")


                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time

                    infer_elapsed_list.append(elapsed)

                    if f1[1] > best_f1:
                        best_epoch = epoch
                        best_precision = precision[1]
                        best_recall = recall[1]
                        best_f1 = f1[1]
                        best_micro_f1 = mf1

                        now_best = {"train_loss": train_loss, "val_loss": val_loss, "precision": precision[1], "recall": recall[1], "F1": f1[1],  "Micro F1:": mf1}
                        print("Now Best:", now_best)



                        best_weights = copy.deepcopy(model.state_dict())
                        torch.save(
                            {'args': args,
                             'state_dict': best_weights},
                            logging_dir + '/' + args.model + '-' + str(t) + '-' + str(k) + '.pth')

                    print(
                        'Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Micro F1: {:.4f}'.format(
                            epoch + 1, train_loss, val_loss, precision[1], recall[1], f1[1], mf1))

                epoch_time_list.append(train_time)

                total_epoch += 1

            total_epoch_list.append(total_epoch)
            total_time = timer() - total_start_time
            total_time_list.append(total_time)

            print('Best Epoch: {:02d}, Best Precision: {:.4f}, Best Recall: {:.4f}, Best F1: {:.4f}, Best Micro F1: {:.4f}'.format(
                    best_epoch + 1,  best_precision, best_recall, best_f1, best_micro_f1))
            test_precision_list.append(best_precision)
            test_recall_list.append(best_recall)
            test_f1_score_list.append(best_f1)
            test_micro_f1_score_list.append(best_micro_f1)


        t_times_test_precision_list.append(test_precision_list)
        t_times_test_recall_list.append(test_recall_list)
        t_times_test_micro_f1_score_list.append(test_micro_f1_score_list)
        t_times_test_f1_score_list.append(test_f1_score_list)


        t_times_total_time_list.append(total_time_list)
        t_times_epoch_time_list.append(epoch_time_list)
        t_times_total_epoch_list.append(total_epoch_list)
        t_times_infer_elapsed_list.append(infer_elapsed_list)

    print("T-times Epoch Time: {} ~ {}".format(np.mean(nested_average_numpy(t_times_epoch_time_list)),
                                               np.std(nested_average_numpy(t_times_epoch_time_list))))
    print("T-times Total Epoch: {} ~ {}".format(np.mean(nested_average_numpy(t_times_total_epoch_list)),
                                                np.std(nested_average_numpy(t_times_total_epoch_list))))
    print("T-times Total Time: {} ~ {}".format(np.mean(nested_average_numpy(t_times_total_time_list)),
                                               np.std(nested_average_numpy(t_times_total_time_list))))

    print("T-times Inference Elapsed: {} ~ {}".format(np.mean(nested_average_numpy(t_times_infer_elapsed_list)),
                                                      np.std(nested_average_numpy(t_times_infer_elapsed_list))))


    print(t_times_test_micro_f1_score_list)
    print(t_times_test_precision_list)
    print(t_times_test_recall_list)
    print(t_times_test_f1_score_list)

    print('T-times cross validation test micro f1 score:{} ~ {}'.format(np.mean(nested_average_numpy(t_times_test_micro_f1_score_list)),
                                                                  np.std(nested_average_numpy(t_times_test_micro_f1_score_list))))
    print('T-times cross validation test precision:{} ~ {}'.format(np.mean(nested_average_numpy(t_times_test_precision_list)),
                                                                  np.std(nested_average_numpy(t_times_test_precision_list))))
    print('T-times cross validation test recall:{} ~ {}'.format(np.mean(nested_average_numpy(t_times_test_recall_list)),
                                                                  np.std(nested_average_numpy(t_times_test_recall_list))))
    print('T-times cross validation test f1_score:{} ~ {}'.format(np.mean(nested_average_numpy(t_times_test_f1_score_list)),
                                                                 np.std(nested_average_numpy(t_times_test_f1_score_list))))

def nested_average_numpy(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=object)

    inner_means = np.array([np.mean(sublist) for sublist in arr])
    return inner_means

if __name__ == "__main__":
    main()