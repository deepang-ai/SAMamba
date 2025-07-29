# -*- coding: utf-8 -*-
import torch
from torch import nn
import copy
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv
from .layers import TransformerEncoderLayer, MambaEncoderLayer
from einops import repeat
import torch_geometric.utils as utils
# from .utils import global_max_pool

from torch_geometric.nn import global_max_pool


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         ) + output
        if self.norm is not None:
            output = self.norm(output)
        return output + x


def _get_clones(module, N):
    return nn.modules.container.ModuleList([copy.deepcopy(module) for i in range(N)])


class GraphMambaEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None, ptr=None,
                batch=None):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         batch=batch
                         ) + output
        if self.norm is not None:
            output = self.norm(output)
        return output + x


class GraphEncoder(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", ie="transformer", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', order='sd', task='graph', **kwargs):
        super().__init__()

        self.d_model = d_model
        self.in_size = in_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.task = task

        self.subgraph_conv = GATConv(d_model, d_model, dropout=dropout,
                                     add_self_loops=True, negative_slope=0.01)

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        self.ie = ie

        if self.ie == "transformer":
            encoder_layer = TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
                gnn_type=gnn_type, se=se, **kwargs)
            self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        else:
            encoder_layer = MambaEncoderLayer(
                d_model, dim_feedforward, dropout, batch_norm=batch_norm,
                gnn_type=gnn_type, se=se, order=order, **kwargs)
            self.encoder = GraphMambaEncoder(encoder_layer, num_layers)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        elif global_pool == 'gat':
            self.pooling = global_max_pool
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None

        output = self.embedding(x.float()) if node_depth is None else self.embedding(x, node_depth.view(-1, ))

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        if self.ie == "transformer":
            node_reps = self.encoder(
                output,
                edge_index,
                complete_edge_index,
                edge_attr=edge_attr,
                degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=data.ptr,
                return_attn=return_attn
            )
        else:
            node_reps = self.encoder(
                output,
                edge_index,
                edge_attr=edge_attr,
                degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=data.ptr,
                batch=data.batch
            )

        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                graph_reps = node_reps[-bsz:]

            elif self.global_pool == 'gat':
                # Subgraph Embedding:
                row = torch.arange(data.batch.size(0)).to("cuda")
                edge_index = torch.stack([row, data.batch], dim=0)
                out = self.pooling(node_reps, data.batch).relu_()
                graph_reps = self.subgraph_conv((node_reps, out), edge_index)
            else:
                graph_reps = self.pooling(node_reps, data.batch)

        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](graph_reps))
            return pred_list

        if self.task == 'graph':
            return node_reps, graph_reps, self.classifier(graph_reps)
        elif self.task == 'node':
            return node_reps, graph_reps, self.classifier(node_reps)
        else:
            assert 0

    def loss_un(self, x, x_aug):  # loss_{self}

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / 0.2)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # 判断分母是否为0
        minnum = 0
        if ((sim_matrix.sum(dim=1) - pos_sim) == 0).any():
            minnum = torch.tensor(1e-6)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + minnum)
        # loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def loss_su(self, pred_out, target):  # loss_{pred}

        loss = torch.nn.CrossEntropyLoss()
        try:
            loss = loss(pred_out, target.squeeze())
        except:
            loss = loss(pred_out, target.squeeze(0))
        return loss

    def loss_cal(self, x, x_aug, pred_out, target, Lambda):
        loss_un = self.loss_un(x, x_aug)
        loss_su = self.loss_su(pred_out, target)
        return loss_su + Lambda * loss_un, loss_un, loss_su


ACT_FUNC = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'mish': nn.Mish(),
    'gelu': nn.GELU(),
    'elu': nn.ELU()
}