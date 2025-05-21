import torch
from torch import nn
import dgl.function as fn
import dgl


class GNN(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 in_edge_dim,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 norm,
                 prelu):
        super(GNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.edge_dim = in_edge_dim

        # 第一层
        self.layers.append(BasicGNNConv(node_dim=in_feats,
                                        edge_dim=in_edge_dim,
                                        out_dim=n_hidden))
        self.norms.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(n_hidden))
        self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        # 中间层
        for _ in range(n_layers - 2):
            self.layers.append(BasicGNNConv(node_dim=n_hidden,
                                            edge_dim=in_edge_dim,
                                            out_dim=n_hidden))
            self.norms.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.99) if norm == 'batch' else \
                                  torch.nn.LayerNorm(n_hidden))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        # 最后一层
        self.layers.append(BasicGNNConv(node_dim=n_hidden,
                                        edge_dim=in_edge_dim,
                                        out_dim=n_classes))
        self.norms.append(torch.nn.BatchNorm1d(n_classes, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(n_classes))
        self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = n_classes

    def forward(self, g, h, return_hidden=False):
        hidden_list = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            # 获取边特征
            edge_attr = g.edata['attr'] if 'attr' in g.edata else \
                torch.zeros((g.number_of_edges(), self.edge_dim), device=h.device)
            h = layer(g, h, edge_attr)
            hidden_list.append(h)
            h = self.activations[i](self.norms[i](h))

        if return_hidden:
            return h, hidden_list
        else:
            return h


class BasicGNNConv(nn.Module):
    """
    基础的GNN卷积层

    Args:
        node_dim (int): 输入节点特征维度
        edge_dim (int): 边特征维度
        out_dim (int): 输出特征维度
        aggregator_type (str): 聚合函数类型
    """

    def __init__(self, node_dim, edge_dim, out_dim, aggregator_type="mean"):
        super(BasicGNNConv, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.aggregator_type = aggregator_type

        # 转换节点特征的线性层
        self.node_transform = nn.Linear(node_dim, out_dim)
        # 转换边特征的线性层
        self.edge_transform = nn.Linear(edge_dim, out_dim)
        # 组合转换后的特征的线性层
        self.combine = nn.Linear(2 * out_dim, out_dim)

    def forward(self, g, node_feat, edge_feat):
        with g.local_scope():
            # 1. 转换节点和边的特征
            transformed_node = self.node_transform(node_feat)
            transformed_edge = self.edge_transform(edge_feat)

            # 2. 存储特征
            g.ndata['h'] = transformed_node
            g.edata['e'] = transformed_edge

            # 3. 消息传递和聚合
            if self.aggregator_type == "mean":
                g.update_all(self.message_func, fn.mean('m', 'agg'))
            elif self.aggregator_type == "sum":
                g.update_all(self.message_func, fn.sum('m', 'agg'))
            elif self.aggregator_type == "max":
                g.update_all(self.message_func, fn.max('m', 'agg'))
            else:
                raise NotImplementedError(f"Aggregator type {self.aggregator_type} not implemented.")

            # 4. 更新节点表示
            agg_feat = g.ndata['agg']
            combined = torch.cat([transformed_node, agg_feat], dim=1)

            return self.combine(combined)

    def message_func(self, edges):
        """
        消息函数：组合源节点特征和边特征
        """
        return {'m': edges.src['h'] + edges.data['e']}
