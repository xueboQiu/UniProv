import dgl
import torch

class GCN(torch.nn.Module):
    # def __init__(self,
    #              in_feats,
    #              hidden_lst,
    #              dropout,
    #              norm,
    #              prelu):
    #     super(GCN, self).__init__()
    #     self.layers = torch.nn.ModuleList()
    #     self.norms = torch.nn.ModuleList()
    #     self.activations = torch.nn.ModuleList()
    #     self.in_feats = in_feats
    #     hidden_lst = [in_feats] + hidden_lst
    #     for in_, out_ in zip(hidden_lst[:-1], hidden_lst[1:]):
    #         self.layers.append(dgl.nn.GraphConv(in_, out_, allow_zero_in_degree=True))
    #         self.norms.append(torch.nn.BatchNorm1d(out_, momentum=0.99) if norm == 'batch' else \
    #                           torch.nn.LayerNorm(out_))
    #         self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
    #
    #     self.dropout = torch.nn.Dropout(p=dropout)
    #     self.n_classes = hidden_lst[-1]
    #
    #     # 参数初始化
    #     for layer in self.layers:
    #         if isinstance(layer, dgl.nn.GraphConv):  # 检查是否是 GraphConv 层
    #             torch.nn.init.xavier_uniform_(layer.weight)  # 对权重进行 Xavier 初始化
    #             if layer.bias is not None:  # 如果有偏置项
    #                 torch.nn.init.zeros_(layer.bias)  # 将偏置项初始化为 0
    def __init__(self,
                 in_feats,
                 n_hidden,  # 隐藏层维度
                 n_classes,  # 输出维度
                 n_layers,  # 层数
                 dropout,
                 norm,
                 prelu):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.n_layers = n_layers

        # 第一层：输入层 -> 隐藏层
        self.layers.append(dgl.nn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
        self.norms.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(n_hidden))
        self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        # 中间层：隐藏层 -> 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(dgl.nn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
            self.norms.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.99) if norm == 'batch' else \
                                  torch.nn.LayerNorm(n_hidden))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        # 最后一层：隐藏层 -> 输出层
        self.layers.append(dgl.nn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.norms.append(torch.nn.BatchNorm1d(n_classes, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(n_classes))
        self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = n_classes

        # 参数初始化
        for layer in self.layers:
            if isinstance(layer, dgl.nn.GraphConv):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, g, features, return_hidden=False):

        # for name, param in self.layers[0].named_parameters():
        #     assert not  torch.isnan(param).any().item(), f"layer0 weight contains NaN, {param}"
        #     assert not torch.isinf(param).any().item(), f"layer0 weight contains Inf!, {param}"

        if type(g) is list:
            h = g[0].ndata['feat']['_N'].to(self.layers[-1].weight.device)
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g[i].to(self.layers[-1].weight.device), h)    # 根据不同层的链接情况进行图卷积
                h = self.activations[i](self.norms[i](h))
        else:
            h = features
            hidden_list = []
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g, h)
                hidden_list.append(h)
                h = self.activations[i](self.norms[i](h))
        if return_hidden:
            return h, hidden_list
        else:
            return h