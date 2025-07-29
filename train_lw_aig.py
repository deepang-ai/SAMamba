# -*- coding: utf-8 -*-
import os
import sys

from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import torch_geometric.utils as utils
from samamba.models import GraphEncoder
from samamba.data import GraphDataset
from samamba.utils import count_parameters
from samamba.gnn_layers import GNN_TYPES
from samamba.utils import add_zeros, extract_node_feature, extract_edge_feature
from timeit import default_timer as timer

from datasets_lw_AIG.lw_AIG_dataset_pyg import PygGraphPropPredDataset
from datasets_lw_AIG.lw_AIG_evaluator import Evaluator

from functools import partial

from utils import ColumnNormalizeFeatures
from utils import EarlyStopping
import torch_geometric.transforms as T
from torch.cuda import memory_allocated, max_memory_allocated, reset_peak_memory_stats
from samamba.position_encoding import POSENCODINGS



from thop import profile
from samamba.utils import Logger


from sklearn.metrics import  f1_score, precision_score, recall_score
from transform import Augmentor_Transform
def load_args():
    parser = argparse.ArgumentParser(
        description='SAMamba',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, help='random seed', default=18)
    parser.add_argument('--model', type=str, default='SAMamba',
                        help='model choices')
    parser.add_argument('--dataset', type=str, default="exchange/Volume",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=2, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=64, help="hidden dimension")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout") #0.6
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')

    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')

    parser.add_argument('--num-class', type=int, default=2,
                        help='Number of classes')

    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=10, help="number of epochs for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', type=bool, default=False, help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=128, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='gat',  # Base GNN model
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=1, help="number of layers for GNNs")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add', 'gat'],
                        # Aggregate node-level representations into a graph representation.
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="khopgnn",  # k-subtree or k-subgraph GNN extractor
                        help='Extractor type: khopgnn, or gnn')

    parser.add_argument('--ie', type=str, default="mamba",
                        help='information encoder type: transformer, or mamba')

    parser.add_argument('--aggr', type=str, default='add',  # Aggregation edge features to obtain nodes initial features
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--not_extract_node_feature', action='store_true')

    parser.add_argument('--profile', action='store_true', help='FLOPs and GPU memory statistics')


    parser.add_argument('--k-ford', '-KF', type=int, help='', default=3)

    parser.add_argument('--early_stop', type=int, help='', default=1)
    parser.add_argument('--early_stop_mindelta', '-min_delta', type=float, help='gpu id', default=-0.)
    parser.add_argument('--patience', type=int, help='patience', default=20)

    parser.add_argument('--training-times', '-t-times', type=int, help='training times', default=3)

    parser.add_argument('--Lambda', type=float, help='loss trade-off', default=0.01)

    parser.add_argument('--order', type=str, default="sd", choices=['nd', 'sd'],
                        help='node prioritization strategy in graph mamba: node degree, or subgraph degree')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    # args.use_cuda = False
    args.batch_norm = not args.layer_norm

    args.save_logs = True
    args.outdir = './outdir'
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    total_loss = 0
    total_loss_pred = 0
    total_loss_self = 0



    tic = timer()

    # for i, (data_v1, data_v2, data_raw) in enumerate(zip(loader_aug1, loader_aug2, loader)):

    for i, data in enumerate(loader):
        data_v1, data_v2, data_raw = data

        if use_cuda:
            data_v1, data_v2, data_raw = data_v1.cuda(), data_v2.cuda(), data_raw.cuda()

        # size = len(data_v1.y)
        if epoch < args.warmup:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)


        optimizer.zero_grad()

        if args.model == "SAMamba" or args.model == "Ethident":
            node_reps_v1, graph_reps_v1, pred_out_v1 = model(data_v1)
            node_reps_v2, graph_reps_v2, pred_out_v2 = model(data_v2)
            node_reps, graph_reps, pred_out = model(data_raw)

            loss, loss_self, loss_pred = model.loss_cal(x=graph_reps_v1, x_aug=graph_reps_v2, pred_out=pred_out,
                                                            target=data_raw.y, Lambda=args.Lambda)

        elif args.model == "AEtransGAT":
            node_reps, graph_reps, pred_out = model.encoder(data_raw)
            loss = torch.nn.CrossEntropyLoss()
            try:
                loss1 = loss(pred_out, data_raw.y.squeeze())
            except:
                loss1 = loss(pred_out, data_raw.y.squeeze(0))
            loss2 = model.recon_loss(node_reps, data_raw.edge_index)
            loss = 0.6 * loss1 + 0.4 * loss2

        else:
            node_reps, graph_reps, pred_out = model(data_raw)
            loss = torch.nn.CrossEntropyLoss()

            try:
                loss = loss(pred_out, data_raw.y.squeeze())
            except:
                loss = loss(pred_out, data_raw.y.squeeze(0))

        loss.backward()
        optimizer.step()
        if args.model == "SAMamba" or args.model == "Ethident":
            total_loss += float(loss) * data_raw.num_graphs
            total_loss_pred += float(loss_pred) * data_raw.num_graphs
            total_loss_self += float(loss_self) * data_raw.num_graphs
        else:
            total_loss += float(loss) * data_raw.num_graphs
            total_loss_pred += float(loss) * data_raw.num_graphs
            total_loss_self += 0

            # running_loss += loss.item() * size


    # flops, params = profile(model, inputs=(data_raw,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')


    toc = timer()
    # n_sample = len(loader.dataset)
    # epoch_loss = running_loss / n_sample
    epoch_loss = total_loss / len(loader.dataset)
    epoch_loss_pred = total_loss_pred / len(loader.dataset)
    epoch_loss_self = total_loss_self / len(loader.dataset)

    time = toc - tic

    print('Train loss: {:.4f};  Loss pred: {:.4f}; Loss self: {:.4f}; time: {:.2f}s'.format(epoch_loss, epoch_loss_pred, epoch_loss_self, toc - tic))
    return epoch_loss, time


def remove_hooks(m):
    if hasattr(m, "_forward_hooks"):
        m._forward_hooks.clear()




def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    y_pred = []
    y_true = []

    tic = timer()
    model.apply(remove_hooks)

    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            if args.model == "AEtransGAT":
                output = model.encoder(data)[2]
            else:
                output = model(data)[2]
            try:
                loss = criterion(output, data.y.squeeze())
            except:
                loss = criterion(output, data.y.squeeze(0))

            y_true.append(data.y.cpu())
            y_pred.append(output.argmax(dim=-1).view(-1, 1).cpu())


            running_loss += loss.item() * size



    toc = timer()


    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()



    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    evaluator = Evaluator(name=args.dataset)

    score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average=None)[1]
    recall = recall_score(y_true, y_pred, average=None)[1]
    accuracy = f1_score(y_true=y_true, y_pred=y_pred, average=None)[1]

    # score = evaluator.eval({'y_true': [y_true],
    #                         'y_pred': [y_pred]})['acc']

    time =  toc - tic

    print('{} loss: {:.4f} score: {:.4f} time: {:.2f}s'.format(
        split, epoch_loss, score, toc - tic))
    return precision, recall, accuracy, score, epoch_loss, time


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
    global args
    args = load_args()

    logging_dir = os.getcwd() + '/outdir/' + args.dataset  + '-' + args.model + '-' + str(datetime.now()).replace(' ', '-')


    Logger(logging_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = './datasets_lw_AIG'
    num_edge_features = 2
    num_node_features = 14885
    input_size = num_node_features

    if args.not_extract_node_feature:
        transform = T.Compose([
            partial(extract_edge_feature, reduce=args.aggr),
            ColumnNormalizeFeatures(['edge_attr']),
            T.NormalizeFeatures()
        ])
        input_size = num_edge_features
    else:
        # functools.partial 这个高阶函数用于部分应用一个函数。部分应用是指，基于一个函数创建一个新的可调用对象，把原函数的某些参数固定。使用这个函数可以把接受一个或多个参数的函数改编成需要回调的 API，这样参数更少
        # https://blog.csdn.net/qsloony/article/details/123802110?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170212950916800197083519%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170212950916800197083519&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-6-123802110-null-null.142^v96^pc_search_result_base1&utm_term=functool%20partial&spm=1018.2226.3001.4187
        transform = T.Compose([
            partial(extract_node_feature, reduce=args.aggr),
            ColumnNormalizeFeatures(['edge_attr']),
            T.NormalizeFeatures()
        ])
        # transform = partial(extract_node_feature, reduce=args.aggr)
        input_size = num_node_features + num_edge_features



    dataset = PygGraphPropPredDataset(name=args.dataset, root=data_path, transform=transform, num_class=args.num_class)

    print(dataset[0])

    transform_aug1 = T.Compose([
        partial(extract_node_feature, reduce=args.aggr),
        Augmentor_Transform['identity'](prob=None),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    dataset_aug1 = PygGraphPropPredDataset(name=args.dataset, root=data_path, transform=transform_aug1, num_class=args.num_class)
    print(dataset_aug1[0])

    transform_aug2 = T.Compose([
        partial(extract_node_feature, reduce=args.aggr),
        Augmentor_Transform['nodeDrop'](prob=0.1),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    dataset_aug2 = PygGraphPropPredDataset(name=args.dataset, root=data_path, transform=transform_aug2, num_class=args.num_class)
    print(dataset_aug2[0])


    all_score_list = []

    t_times_test_precision_list = []
    t_times_test_recall_list = []
    t_times_test_micro_f1_score_list = []
    t_times_test_f1_score_list = []

    t_times_infer_elapsed_list = []
    t_times_time_per_graph_list = []
    t_times_speed_list = []

    t_times_total_time_list = []
    t_times_epoch_time_list = []
    t_times_total_epoch_list = []

    seeds = [i for i in range(args.training_times)]

    for t in range(args.training_times):
        print("========================training times:{}========================".format(t))


        test_precision_list = []
        test_recall_list = []
        test_micro_f1_score_list = []
        test_f1_score_list = []

        infer_elapsed_list = []
        time_per_graph_list = []
        speed_list = []

        total_time_list = []
        epoch_time_list = []
        total_epoch_list = []



        for k in range(args.k_ford):
            print("========================k_idx:{}========================".format(k))



            split_idx = dataset.get_idx_split(X=np.arange(len(dataset)),
                                              Y=np.array([dataset[i].y.item() for i in range(len(dataset))]),
                                              seed=seeds[t], K=args.k_ford, k_idx=k, dataset=args.dataset)




            train_dset_aug1 = GraphDataset(dataset_aug1[split_idx['train']], degree=True,
                                      k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
                                      return_complete_index=False)
            train_dset_aug2 = GraphDataset(dataset_aug1[split_idx['train']], degree=True,
                                      k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
                                      return_complete_index=False)

            train_dset = GraphDataset(dataset[split_idx['train']], degree=True,
                                      k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
                                      return_complete_index=False)

            # Data(edge_index=[2, 4408], edge_attr=[4408, 7], y=[1, 1], num_nodes=300, x=[300, 7], degree=[300])
            ##

            val_dset = GraphDataset(dataset[split_idx['valid']], degree=True,
                                    k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
                                    return_complete_index=False)
            # train_loader_aug1 = DataLoader(train_dset_aug1, batch_size=args.batch_size, shuffle=False)
            # train_loader_aug2 = DataLoader(train_dset_aug2,
            #                                batch_size=args.batch_size, shuffle=False)
            # train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False)
            val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

            train_loader = DataLoader(list(zip(train_dset_aug1, train_dset_aug2, train_dset)),
                                      batch_size=args.batch_size, shuffle=True)

            abs_pe_encoder = None
            if args.abs_pe and args.abs_pe_dim > 0:
                abs_pe_method = POSENCODINGS[args.abs_pe]
                abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
                if abs_pe_encoder is not None:
                    abs_pe_encoder.apply_to(train_dset_aug1)
                    abs_pe_encoder.apply_to(train_dset_aug2)
                    abs_pe_encoder.apply_to(train_dset)
                    abs_pe_encoder.apply_to(val_dset)




            if 'pna' in args.gnn_type or args.gnn_type == 'mpnn':
                deg = torch.cat([
                    utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dset])
            else:
                deg = None



            model = GraphEncoder(in_size=input_size,
                                     num_class=dataset.num_classes,
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
                                     order=args.order
                                     )





            print("Model: {}".format(args.model))
            print(model)

            if args.use_cuda:
                model.cuda()


            if args.profile:

                loader = DataLoader(train_dset, batch_size=1, shuffle=False)

                dummy_data = next(iter(loader)).cuda() if args.use_cuda else next(iter(loader))

                flops = get_flops(model, dummy_data)
                print(f"Model {args.model} Statistic：")
                print(f"FLOPs: {flops / 1e9:.8f} G")

                print("Total number of parameters: {} M".format(count_parameters(model) / 1e6))


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)


            lr_steps = args.lr / (args.warmup * len(train_loader))

            def warmup_lr_scheduler(s):
                lr = s * lr_steps
                return lr

            test_dset = GraphDataset(dataset[split_idx['test']], degree=True,
                                     k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
                                     return_complete_index=False)
            test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)


            if abs_pe_encoder is not None:
                abs_pe_encoder.apply_to(test_dset)



            print(test_loader)


            print("Training...")
            best_val_loss = float('inf')
            best_val_score = 0
            best_model = None
            best_epoch = 0
            logs = defaultdict(list)
            start_time = timer()

            early_stopping = EarlyStopping(patience=args.patience, min_delta=args.early_stop_mindelta)

            total_epoch = 0

            # ----------------------
            # GPU Memory before training
            # ----------------------
            if args.use_cuda and args.profile:
                print(f"GPU device: {torch.cuda.get_device_name(0)}")
                mem_before = get_gpu_memory()
                print(
                    f"GPU Memory before training: allocated {mem_before['allocated']:.2f} MB, Max {mem_before['max_allocated']:.2f} MB")

            for epoch in range(args.epochs):


                print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))


                if args.use_cuda and args.profile:
                    reset_peak_memory_stats()
                # train_loss, train_time = train_epoch(model, train_loader_aug1, train_loader_aug2, train_loader, criterion, optimizer, warmup_lr_scheduler, epoch, args.use_cuda)
                train_loss, train_time = train_epoch(model, train_loader, criterion, optimizer, warmup_lr_scheduler, epoch, args.use_cuda)
                # 训练后获取内存使用
                if args.use_cuda and args.profile:
                    mem_train = get_gpu_memory()
                    print(
                        f"Epoch {epoch + 1} GPU Memory after training: allocated {mem_train['allocated']:.2f} MB, Max {mem_train['max_allocated']:.2f} MB")

                val_precision, val_recall, val_accuracy, val_score, val_loss, val_time = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')

                import time
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                if args.use_cuda and args.profile:
                    reset_peak_memory_stats()
                test_precision, test_recall, test_accuracy, test_score, test_loss, test_time = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')
                if args.use_cuda and args.profile:
                    mem_infer = get_gpu_memory()
                    print(
                        f"Epoch {epoch + 1} GPU Memory after training: allocated {mem_infer['allocated']:.2f} MB, Max {mem_infer['max_allocated']:.2f} MB")

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                infer_elapsed_list.append(elapsed)
                time_per_graph_list.append(elapsed / len(test_loader.dataset))
                speed_list.append(len(test_loader.dataset) /elapsed)


                epoch_time_list.append(train_time + val_time + test_time)


                total_epoch += 1

                if epoch >= args.warmup:
                    lr_scheduler.step()

                logs['train_loss'].append(train_loss)
                logs['val_precision'].append(val_precision)
                logs['val_recall'].append(val_recall)
                logs['val_accuracy'].append(val_accuracy)
                logs['val_score'].append(val_score)
                logs['test_precision'].append(test_precision)
                logs['test_recall'].append(test_recall)
                logs['test_accuracy'].append(test_accuracy)
                logs['test_score'].append(test_score)
                if val_score > best_val_score:
                    best_val_precision = val_precision
                    best_val_recall = val_recall
                    best_val_score = val_score
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_weights = copy.deepcopy(model.state_dict())

                if args.early_stop:
                    early_stopping(val_loss, results=[epoch, train_loss, val_precision, val_recall, val_accuracy, val_score, val_loss, test_precision, test_recall, test_accuracy, test_score, test_loss])
                    if early_stopping.early_stop:
                        print('\n=====final results=====')
                        _epoch, _train_loss, _val_precision, _val_recall, _val_accuracy, _val_score, _val_loss, _test_precision, _test_recall, _test_accuracy, _test_score, _test_loss = early_stopping.best_results
                        all_score_list.append(_test_score)
                        test_precision_list.append(_test_precision)
                        test_recall_list.append(_test_recall)
                        test_f1_score_list.append(_test_accuracy)
                        test_micro_f1_score_list.append(_test_score)
                        print(f'Exp: {1},  Epoch: {_epoch:03d},   '
                              f'Train_Loss: {_train_loss:.4f},   '
                              f'Val_Loss: {_val_loss:.4f},   '
                              f'Val_Precision: {_val_precision:.4f},   '
                              f'Val_Recall: {_val_recall:.4f},   '
                              f'Val_accuracy: {_val_accuracy:.4f},   '
                              f'Val_Score: {_val_score:.4f},   '
                              f'Val_Loss: {_val_loss:.4f},   '
                              f'Test_Precision: {_test_precision:.4f},   '
                              f'Test_Recall: {_test_recall:.4f},   '
                              f'Test_accuracy: {_test_accuracy:.4f},   '
                              f'Test_Score: { _test_score:.4f},   '
                              f'Test_loss: {_test_loss:.4f}\n\n'
                              )
                        break
                else:
                    all_score_list.append(test_score)


            print(infer_elapsed_list)
            print(time_per_graph_list)
            print(speed_list)
            print("Elapsed: {}~{}".format(np.mean(infer_elapsed_list), np.std(infer_elapsed_list)))
            print("Time per graph: {}~{}".format(np.mean(time_per_graph_list), np.std(time_per_graph_list)))
            print("Speed: {}~{}".format(np.mean(speed_list), np.std(speed_list)))

            total_epoch_list.append(total_epoch)
            total_time = timer() - start_time

            total_time_list.append(total_time)

            print("Total Time: {:.4f}".format(total_time))

            print("best val loss: {} test_score: {:.4f}".format(_val_loss , _test_score))
            model.load_state_dict(best_weights, strict=False)

            print()
            print("Testing...")
            test_precision, test_recall, test_accuracy, test_score, test_loss, test_time = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

            print("test Score {:.4f}".format(test_score))

            if args.save_logs:
                logs = pd.DataFrame.from_dict(logs)
                logs.to_csv(args.outdir + '/logs.csv')
                results = {
                    'test_score': test_score,
                    'test_loss': test_loss,
                    'val_score': best_val_score,
                    'val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'total_time': total_time,
                }
                results = pd.DataFrame.from_dict(results, orient='index')
                results.to_csv(args.outdir + '/results.csv',
                               header=['value'], index_label='name')
                torch.save(
                    {'args': args,
                     'state_dict': best_weights},
                    args.outdir + '/' + args.model + '-' + str(t) +  '-' + str(k) + '.pth')

            print("Epoch Time List:", epoch_time_list)
            print("Total Epoch List:", total_epoch_list)
            print("Total Time List:", total_time_list)

        t_times_test_precision_list.append(test_precision_list)
        t_times_test_recall_list.append(test_recall_list)
        t_times_test_micro_f1_score_list.append(test_micro_f1_score_list)
        t_times_test_f1_score_list.append(test_f1_score_list)

        t_times_infer_elapsed_list.append(infer_elapsed_list)
        t_times_time_per_graph_list.append(time_per_graph_list)
        t_times_speed_list.append(speed_list)

        t_times_total_time_list.append(total_time_list)
        t_times_epoch_time_list.append(epoch_time_list)
        t_times_total_epoch_list.append(total_epoch_list)





    print("T-times Epoch Time: {} ~ {}".format(np.mean(nested_average_numpy(t_times_epoch_time_list)), np.std(nested_average_numpy(t_times_epoch_time_list))))
    print("T-times Total Epoch: {} ~ {}".format(np.mean(nested_average_numpy(t_times_total_epoch_list)), np.std(nested_average_numpy(t_times_total_epoch_list))))
    print("T-times Total Time: {} ~ {}".format(np.mean(nested_average_numpy(t_times_total_time_list)), np.std(nested_average_numpy(t_times_total_time_list))))

    print("T-times Inference Elapsed: {} ~ {}".format(np.mean(nested_average_numpy(t_times_infer_elapsed_list)), np.std(nested_average_numpy(t_times_infer_elapsed_list))))
    print("T-times Time Per Graph: {} ~ {}".format(np.mean(nested_average_numpy(t_times_time_per_graph_list)), np.std(nested_average_numpy(t_times_time_per_graph_list))))
    print("T-times Speed: {} ~ {}".format(np.mean(nested_average_numpy(t_times_speed_list)), np.std(nested_average_numpy(t_times_speed_list))))

    print('T-times cross validation test micro f1 score:{} ~ {}'.format(np.mean(nested_average_numpy(t_times_test_f1_score_list)),
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