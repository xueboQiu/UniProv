import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np


class GraphSAGEConv(nn.Module):
    """
    GraphSAGE卷积层，包含邻居采样

    Args:
        node_dim (int): 节点特征维度
        edge_dim (int): 边特征维度
        emb_dim (int): 输出嵌入维度
        num_samples (int): 采样邻居数量
        aggr (str): 聚合函数类型，默认为"mean"
    """

    def __init__(self, node_dim, edge_dim, emb_dim, num_samples=10, aggr="mean"):
        super(GraphSAGEConv, self).__init__()
        self.linear = nn.Linear(node_dim * 2, emb_dim)  # 2倍是因为拼接
        # 添加边特征转换层，将边特征转换为与节点特征相同的维度
        self.edge_linear = nn.Linear(edge_dim, node_dim)
        self.edge_dim = edge_dim
        self.num_samples = num_samples
        self.aggr = aggr

    def forward(self, g, node_feat, edge_feat):
        with g.local_scope():
            # 1. 对每个节点采样固定数量的邻居
            sampled_g = dgl.sampling.sample_neighbors(g,
                                                      g.nodes(),
                                                      self.num_samples,
                                                      replace=True)  # 允许重复采样

            # 2. 获取采样后图的边特征
            if edge_feat is not None:
                edge_feat = edge_feat[sampled_g.edata[dgl.EID]]

            # 3. 存储特征
            sampled_g.ndata['h'] = node_feat
            if edge_feat is not None:
                sampled_g.edata['e'] = edge_feat

            # 4. 消息传递和聚合（只在采样的邻居中进行）
            if self.aggr == "mean":
                sampled_g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            elif self.aggr == "max":
                sampled_g.update_all(self.message_func, fn.max('m', 'h_neigh'))
            elif self.aggr == "sum":
                sampled_g.update_all(self.message_func, fn.sum('m', 'h_neigh'))

            # 5. 拼接并更新节点表示
            h_neigh = sampled_g.ndata['h_neigh']
            h_concat = torch.cat([node_feat, h_neigh], dim=1)

            # 6. 线性变换和激活
            return torch.relu(self.linear(h_concat))

    def message_func(self, edges):
        """消息函数：如果有边特征，则组合源节点特征和边特征"""
        if 'e' in edges.data:
            # 将边特征转换为与节点特征相同的维度
            transformed_edge = self.edge_linear(edges.data['e'])
            return {'m': edges.src['h'] + transformed_edge}
        return {'m': edges.src['h']}


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 in_edge_dim,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_samples,  # 每层的采样数
                 dropout=0.0,
                 norm='batch',
                 prelu=False):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.n_layers = n_layers
        self.edge_dim = in_edge_dim

        # 确保num_samples是列表，长度等于层数
        if isinstance(num_samples, int):
            num_samples = [num_samples] * n_layers
        assert len(num_samples) == n_layers, "num_samples长度必须等于层数"

        # 第一层
        self.layers.append(GraphSAGEConv(
            node_dim=in_feats,
            edge_dim=in_edge_dim,
            emb_dim=n_hidden,
            num_samples=num_samples[0]
        ))
        self.norms.append(nn.BatchNorm1d(n_hidden) if norm == 'batch' else nn.LayerNorm(n_hidden))
        self.activations.append(nn.PReLU() if prelu else nn.ReLU())

        # 中间层
        for i in range(n_layers - 2):
            self.layers.append(GraphSAGEConv(
                node_dim=n_hidden,
                edge_dim=in_edge_dim,
                emb_dim=n_hidden,
                num_samples=num_samples[i + 1]
            ))
            self.norms.append(nn.BatchNorm1d(n_hidden) if norm == 'batch' else nn.LayerNorm(n_hidden))
            self.activations.append(nn.PReLU() if prelu else nn.ReLU())

        # 最后一层
        self.layers.append(GraphSAGEConv(
            node_dim=n_hidden,
            edge_dim=in_edge_dim,
            emb_dim=n_classes,
            num_samples=num_samples[-1]
        ))
        self.norms.append(nn.BatchNorm1d(n_classes) if norm == 'batch' else nn.LayerNorm(n_classes))
        self.activations.append(nn.PReLU() if prelu else nn.ReLU())

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h, return_hidden=False):
        hidden_list = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)

            # 获取边特征
            edge_attr = g.edata.get('attr', None)
            if edge_attr is None:
                edge_attr = torch.zeros((g.number_of_edges(), self.edge_dim),
                                        device=h.device)

            h = layer(g, h, edge_attr)
            hidden_list.append(h)
            h = self.activations[i](self.norms[i](h))

        if return_hidden:
            return h, hidden_list
        return h
