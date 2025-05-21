import torch
from torch import nn
import dgl.function as fn
import dgl

class GIN(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 in_edge_dim,
                 n_hidden,  # 隐藏层维度
                 n_classes,  # 输出维度
                 n_layers,  # 层数
                 dropout,
                 norm,
                 prelu):  # 添加边特征维度参数
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.edge_dim = in_edge_dim

        # 第一层：输入层 -> 隐藏层
        self.layers.append(GINConv(node_dim=in_feats,
                                   edge_dim=in_edge_dim,
                                   emb_dim=n_hidden,
                                   input_layer=True))
        self.norms.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(n_hidden))
        self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        # 中间层：隐藏层 -> 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(GINConv(node_dim=n_hidden,
                                       edge_dim=in_edge_dim,
                                       emb_dim=n_hidden))
            self.norms.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.99) if norm == 'batch' else \
                                  torch.nn.LayerNorm(n_hidden))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        # 最后一层：隐藏层 -> 输出层
        self.layers.append(GINConv(node_dim=n_hidden,
                                   edge_dim=in_edge_dim,
                                   emb_dim=n_classes))
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
            # 获取边信息
            edge_attr = g.edata['attr'] if 'attr' in g.edata else \
                torch.zeros((g.number_of_edges(), self.edge_dim),
                            device=h.device)
            h = layer(g, h, edge_attr)
            hidden_list.append(h)
            h = self.activations[i](self.norms[i](h))

        if return_hidden:
            return h, hidden_list
        else:
            return h

class GINConv(nn.Module):
    """
    DGL版本的GIN卷积层

    Args:
        node_dim (int): 节点特征维度
        edge_dim (int): 边特征维度
        emb_dim (int): 输出嵌入维度
        aggregator_type (str): 聚合函数类型，默认为"sum"
        input_layer (bool): 是否为输入层
    """

    def __init__(self, node_dim, edge_dim, emb_dim, aggregator_type="sum", input_layer=False):
        super(GINConv, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.emb_dim = emb_dim
        self.aggregator_type = aggregator_type

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, emb_dim),
            nn.ReLU()
        )

        # 用于自环边的特征
        self.register_buffer('self_loop_attr', torch.zeros(1, edge_dim))
        self.self_loop_attr[:, -1] = 1  # 最后一维设为1表示自环

    def forward(self, g, node_feat, edge_feat):
        with g.local_scope():
            # 1. 添加自环
            g = dgl.add_self_loop(g)

            # 2. 扩展边特征以包含自环
            num_self_loops = g.number_of_nodes()
            num_orig_edges = g.number_of_edges() - num_self_loops
            self_loop_attr = self.self_loop_attr.repeat(num_self_loops, 1)
            if edge_feat is not None:
                edge_feat = torch.cat([edge_feat, self_loop_attr], dim=0)
            else:
                edge_feat = torch.zeros(num_orig_edges, self.edge_dim, device=node_feat.device)
                edge_feat = torch.cat([edge_feat, self_loop_attr], dim=0)

            # 3. 存储特征
            g.ndata['h'] = node_feat
            g.edata['e'] = edge_feat

            # 4. 消息传递
            g.apply_edges(self.edge_func)

            # 5. 聚合
            if self.aggregator_type == "sum":
                g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            else:
                raise NotImplementedError(f"Aggregator type {self.aggregator_type} not implemented.")

            # 6. 更新节点表示
            h_neigh = g.ndata['h_neigh']
            return self.mlp(h_neigh)

    def edge_func(self, edges):
        """边的消息函数"""
        return {'m': torch.cat([edges.data['e'], edges.src['h']], dim=1)}

    def message_func(self, edges):
        """消息函数"""
        return {'m': edges.data['m']}