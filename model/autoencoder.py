import math
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from datas import dgl_graph_copy
from .graphsage import GraphSAGE
from utils import utils
from .gat import GAT
from utils.utils import create_norm, create_similarity_based_mask, calculate_node_overlap, calculate_overlap_matrix
from functools import partial
from itertools import chain

from .gin import GIN
from .gnn import GNN
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random
import numpy as np

def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l,
        args=args
    )
    return model
class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class BayesianLossWeighting(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        # 初始化损失权重的均值和标准差参数
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        # 计算贝叶斯权重
        weights = torch.exp(-self.log_vars)
        # 使用不确定性加权的损失
        weighted_losses = weights * losses + self.log_vars
        return weighted_losses.mean(), weights
class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2, args=None):
        super(GMAEModel, self).__init__()

        self.args = args

        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.edge_recon_fc.apply(init_weights)

        if args.gnn == 'gat':
            dec_in_dim = self.create_gat(activation, e_dim, feat_drop, hidden_dim, n_dim, n_heads, n_layers,
                                         negative_slope, norm, residual)
        elif args.gnn == 'gcn':
            dec_in_dim = self.create_gcn(feat_drop, hidden_dim, n_dim, n_layers, norm)
        elif args.gnn == 'gin':
            dec_in_dim = self.create_gin(e_dim, feat_drop, hidden_dim, n_dim, n_layers, norm)
        elif args.gnn == 'gnn':
            dec_in_dim = self.create_gnn(e_dim, feat_drop, hidden_dim, n_dim, n_layers, norm)
        elif args.gnn == 'graphsage':
            dec_in_dim = self.create_sage(e_dim, feat_drop, hidden_dim, n_dim, n_layers, norm)
        else:
            raise NotImplementedError

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        # discriminator for ming
        self.discriminator = Discriminator(hidden_dim)
        # head for minsg
        self.minsg_head = torch.nn.Sequential(torch.nn.Linear(hidden_dim, self.args.predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.args.predictor_dim, hidden_dim))
        self.minsg_head.apply(init_weights)
        self.discriminator.apply(init_weights)

        self.bayes_weighting = BayesianLossWeighting(num_losses=len(args.tasks))

    def create_gcn(self, feat_drop, hidden_dim, n_dim, n_layers, norm):
        enc_num_hidden = hidden_dim
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        # build encoder
        from model.gcn import GCN
        self.encoder = GCN(
            n_dim,
            enc_num_hidden,
            enc_num_hidden,
            n_layers,
            feat_drop,
            create_norm(norm),
            True,
        )
        # build decoder for attribute prediction
        self.decoder = GCN(
            dec_in_dim,
            dec_num_hidden,
            n_dim,
            1,
            feat_drop,
            create_norm(norm),
            False,
        )
        return dec_in_dim
    def create_sage(self, e_dim, feat_drop, hidden_dim, n_dim, n_layers, norm):
        enc_num_hidden = hidden_dim
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        # build encoder
        from model.gcn import GCN
        self.encoder = GraphSAGE(
            n_dim,
            e_dim,
            enc_num_hidden,
            enc_num_hidden,
            n_layers,
            10,
            feat_drop,
            create_norm(norm),
            True,
        )
        # build decoder for attribute prediction
        self.decoder = GraphSAGE(
            dec_in_dim,
            e_dim,
            dec_num_hidden,
            n_dim,
            1,
            10,
            feat_drop,
            create_norm(norm),
            False,
        )
        return dec_in_dim
    def create_gin(self, e_dim, feat_drop, hidden_dim, n_dim, n_layers, norm):
        enc_num_hidden = hidden_dim
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        # build encoder
        from model.gcn import GCN
        self.encoder = GIN(
            n_dim,
            e_dim,
            enc_num_hidden,
            enc_num_hidden,
            n_layers,
            feat_drop,
            create_norm(norm),
            True,
        )
        # build decoder for attribute prediction
        self.decoder = GIN(
            dec_in_dim,
            e_dim,
            dec_num_hidden,
            n_dim,
            1,
            feat_drop,
            create_norm(norm),
            False,
        )
        return dec_in_dim
    def create_gnn(self, e_dim, feat_drop, hidden_dim, n_dim, n_layers, norm):
        enc_num_hidden = hidden_dim
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        # build encoder
        from model.gcn import GCN
        self.encoder = GNN(
            n_dim,
            e_dim,
            enc_num_hidden,
            enc_num_hidden,
            n_layers,
            feat_drop,
            create_norm(norm),
            True,
        )
        # build decoder for attribute prediction
        self.decoder = GNN(
            dec_in_dim,
            e_dim,
            dec_num_hidden,
            n_dim,
            1,
            feat_drop,
            create_norm(norm),
            False,
        )
        return dec_in_dim
    def create_gat(self, activation, e_dim, feat_drop, hidden_dim, n_dim, n_heads, n_layers, negative_slope, norm,
                   residual):
        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        # build encoder
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )
        # build decoder for attribute prediction
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )
        return dec_in_dim

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)    # functools.partial 是 Python 标准库中的一个工具，用于固定函数的一部分参数。
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):

        # ==================================================================
        total_loss = {}

        if 'p_recon' in self.args.tasks:
            enc_rep, recon_loss = self.node_recon(g)
            total_loss['recon_loss'] = recon_loss

        if 'p_link' in self.args.tasks:
            link_loss = self.link_pred(enc_rep, g)
            total_loss['link_loss'] = link_loss

        if 'p_ming' in self.args.tasks:
            ming_loss = self.ming(g)
            total_loss['ming_loss'] = ming_loss

        if 'p_misgsg' in self.args.tasks:
            minsg_loss = self.misgsg(g)
            total_loss['misgsg_loss'] = minsg_loss

        if 'p_decor' in self.args.tasks:
            decor_loss = self.decor(g)
            total_loss['decor_loss'] = decor_loss

        if 'p_minn' in self.args.tasks:
            gm_loss = self.minn(g)
            total_loss['minn_loss'] = gm_loss

        return total_loss
        # ==================================================================
        # Feature Reconstruction
        # pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        # pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        # use_g = pre_use_g
        # enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        # enc_rep = torch.cat(all_hidden, dim=1)
        # rep = self.encoder_to_decoder(enc_rep)
        #
        # recon = self.decoder(pre_use_g, rep)
        # x_init = g.ndata['attr'][mask_nodes]
        # x_rec = recon[mask_nodes]
        # loss =  torch.tensor([0],dtype=torch.float).to(pre_use_g.device)
        #
        # # ================================================================================
        # loss += self.criterion(x_rec, x_init)
        #
        # # Structural Reconstruction
        # threshold = min(10000, g.num_nodes())
        #
        # negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        # positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        # positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        # sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        # sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        # y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        # y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
        #     g.device)
        #
        # loss += self.recon_loss(y_pred, y)
        # return loss
    def node_recon(self, g):
        # Feature Reconstruction
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        return enc_rep, self.criterion(x_rec, x_init)
    def link_pred(self, enc_rep, g):
        # Structural Reconstruction
        threshold = min(10000, g.num_nodes())
        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
            g.device)
        return self.recon_loss(y_pred, y)
    def misgsg(self, g):
        # 创建采样器和dataloader
        sampler = dgl.dataloading.MultiLayerNeighborSampler([8, 8])

        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        graphs_v1 = []
        # 使用示例
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            # blocks是采样得到的子图列表, blocks[-1]是最后一层的子图
            block = blocks[0].to(self.args.device)  # 2跳邻居子图
            src, dst = block.edges()
            subgraph = dgl.graph((src, dst), num_nodes=block.num_src_nodes())
            subgraph.ndata['attr'] = block.srcdata['attr']
            subgraph.edata['attr'] = block.edata['attr']
            graphs_v1.append(subgraph)

            if len(graphs_v1) == self.args.batch_size: break

        # gm_augmentations = [
        #     dgl.transforms.DropEdge(self.args.gm_edge_drop_ratio), \
        #     dgl.transforms.DropNode(self.args.gm_node_drop_ratio), \
        #     dgl.transforms.FeatMask(self.args.gm_feat_drop_ratio, ['attr'])
        #     ]
        # probabilities = [0.2, 0.3, 0.5]  # 各增强方法的选择概率
        # aug_type = np.random.choice(len(gm_augmentations),
        #                             self.args.batch_size,
        #                             replace=True,
        #                             p=probabilities)
        gm_augmentations = [
            dgl.transforms.FeatMask(self.args.gm_feat_drop_ratio, ['attr'])
            ]
        aug_type = np.random.choice(len(gm_augmentations),
                                    self.args.batch_size,
                                    replace=True)
        aug_type = {k: aug_type[k] for k in range(self.args.batch_size)}

        graphs_v2 = [dgl_graph_copy(g) for g in graphs_v1]
        for i, g in enumerate(graphs_v2): gm_augmentations[aug_type[i]](g)

        # 批处理所有图
        bg1 = dgl.batch(graphs_v1).to(self.args.device)  # 正样本1
        bg2 = dgl.batch(graphs_v2).to(self.args.device)  # 正样本2
        # 提取嵌入
        bg1.ndata['h'] = self.encoder(bg1, bg1.ndata['attr'])
        bg2.ndata['h'] = self.encoder(bg2, bg2.ndata['attr'])
        # 使用均值池化
        z1 = dgl.mean_nodes(bg1, 'h')
        z2 = dgl.mean_nodes(bg2, 'h')
        # 对特征进行L2归一化
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # 计算相似度矩阵 [B, B]
        logits_12 = torch.mm(z1_norm, z2_norm.t())
        logits_21 = torch.mm(z2_norm, z1_norm.t())
        # 创建标签：对角线位置为正样本
        labels = torch.arange(z1_norm.shape[0], device=z1_norm.device)
        # 使用CrossEntropyLoss
        return (nn.CrossEntropyLoss()(logits_12, labels) + nn.CrossEntropyLoss()(logits_21, labels)) /2
    # def gm(self, g):
    #     # gm
    #     gm_sampler = dgl.dataloading.SAINTSampler('node', budget=self.args.sub_size)  # 采样模式为节点采样，budget为包含的节点或边的数量
    #
    #     gm_augmentations = [dgl.transforms.DropEdge(self.args.gm_edge_drop_ratio), \
    #                         # dgl.transforms.DropNode(self.args.gm_node_drop_ratio), \
    #                         # dgl.transforms.FeatMask(self.args.gm_feat_drop_ratio, ['attr'])
    #                         ]
    #     # gm, batch_size: the number of subgraphs contrasted
    #     graphs_v1 = [gm_sampler.sample(g, 0) for _ in range(self.args.batch_size)]
    #     aug_type = np.random.choice(len(gm_augmentations), self.args.batch_size, replace=True)
    #     aug_type = {k: aug_type[k] for k in range(self.args.batch_size)}
    #     graphs_v2 = [dgl_graph_copy(g) for g in graphs_v1]
    #     for i, g in enumerate(graphs_v1): gm_augmentations[aug_type[i]](g)
    #     for i, g in enumerate(graphs_v2): gm_augmentations[aug_type[i]](g)
    #     # # 计算所有图对之间的重叠度
    #     # overlap_matrix = calculate_overlap_matrix(graphs_v1)
    #     # # 为每个正样本选择负样本
    #     # graphs_neg = []
    #     # for i in range(len(graphs_v1)):
    #     #     # 获取与当前图的所有重叠度
    #     #     overlaps = overlap_matrix[i]
    #     #
    #     #     # 找到所有符合条件的候选负样本索引（重叠度<0.5）
    #     #     valid_neg_indices = np.where((overlaps < 0.5) & (np.arange(len(overlaps)) != i))[0]
    #     #
    #     #     if len(valid_neg_indices) > 0:
    #     #         # 如果有符合条件的负样本，随机选择一个
    #     #         neg_idx = np.random.choice(valid_neg_indices)
    #     #     else:
    #     #         # 如果没有符合条件的，选择重叠度最小的（除了自身）
    #     #         valid_indices = np.where(np.arange(len(overlaps)) != i)[0]
    #     #         neg_idx = valid_indices[np.argmin(overlaps[valid_indices])]
    #     #
    #     #     graphs_neg.append(dgl_graph_copy(graphs_v1[neg_idx]))
    #     # 批处理所有图
    #     bg1 = dgl.batch(graphs_v1).to(self.args.device)  # 正样本1
    #     bg2 = dgl.batch(graphs_v2).to(self.args.device)  # 正样本2
    #     # bg_neg = dgl.batch(graphs_neg).to(self.args.device)  # 负样本
    #     # 提取嵌入
    #     bg1.ndata['h'] = self.encoder(bg1, bg1.ndata['attr'])
    #     bg2.ndata['h'] = self.encoder(bg2, bg2.ndata['attr'])
    #     # bg_neg.ndata['h'] = self.encoder(bg_neg, bg_neg.ndata['attr'])
    #     # 使用均值池化
    #     z1 = dgl.mean_nodes(bg1, 'h')
    #     z2 = dgl.mean_nodes(bg2, 'h')
    #     # z_neg = dgl.mean_nodes(bg_neg, 'h')
    #     # 对特征进行L2归一化
    #     z1_norm = F.normalize(z1, dim=1)
    #     z2_norm = F.normalize(z2, dim=1)
    #     # z_neg_norm = F.normalize(z_neg, dim=1)
    #     # 计算正样本对的相似度
    #     # pos_sim = torch.sum(z1_norm * z2_norm, dim=1) / 0.4
    #     # 计算与负样本的相似度
    #     # neg_sim = torch.mm(z1_norm, z_neg_norm.t()) / 0.4
    #
    #     # return -torch.log( torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=1))).mean(
    #
    #     # 计算相似度矩阵 [B, B]
    #     logits_12 = torch.mm(z1_norm, z2_norm.t()) / 0.4  # temperature=0.4
    #     logits_21 = torch.mm(z2_norm, z1_norm.t()) / 0.4  # temperature=0.4
    #     # 创建标签：对角线位置为正样本
    #     labels = torch.arange(z1_norm.shape[0], device=z1_norm.device)
    #     # 使用CrossEntropyLoss
    #     return (nn.CrossEntropyLoss()(logits_12, labels) + nn.CrossEntropyLoss()(logits_21, labels)) /2

    def ming(self, g):
        node_idx = np.random.choice(g.number_of_nodes(), self.args.batch_size, replace=False)
        g = dgl.khop_in_subgraph(g, node_idx, k=self.args.khop_ming)[0]
        X = g.ndata['attr']
        perm = torch.randperm(X.shape[0])
        positive = self.encoder(g, X)
        negative = self.encoder(g, X[perm])
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        l1 = F.binary_cross_entropy(torch.sigmoid(positive), torch.ones_like(positive))
        l2 = F.binary_cross_entropy(torch.sigmoid(negative), torch.zeros_like(negative))
        return l1+l2

    def minn(self, g):
        node_idx = np.random.choice(g.number_of_nodes(), self.args.batch_size, replace=False)
        graphs_v1, center_nodes = dgl.khop_in_subgraph(g, node_idx, k=self.args.khop_minsg)
        g1 = graphs_v1.to(self.args.device)
        g2 = dgl_graph_copy(g1)
        # minsg_augmentations = [
        #    dgl.transforms.FeatMask(self.args.minsg_dfr, ['attr'])
        # ]
        minsg_augmentations = [
            dgl.transforms.DropEdge(self.args.minsg_der), \
           dgl.transforms.FeatMask(self.args.minsg_dfr, ['attr'])
        ]
        for aug in minsg_augmentations: aug(g2)
        def get_loss(h1, h2, temperature):
            f = lambda x: torch.exp(x / temperature)
            refl_sim = f(utils.sim_matrix(h1, h1))        # intra-view pairs
            between_sim = f(utils.sim_matrix(h1, h2))     # inter-view pairs
            x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
            loss = -torch.log(between_sim.diag() / x1)
            return loss
        h1_all = self.minsg_head(self.encoder(g1, g1.ndata['attr']))
        h2_all = self.minsg_head(self.encoder(g2, g2.ndata['attr']))

        # 只取中心节点的表示进行对比学习
        h1 = h1_all[center_nodes]
        h2 = h2_all[center_nodes]

        # l1 = get_loss(h1, h2, self.args.temperature_minsg)
        # l2 = get_loss(h2, h1, self.args.temperature_minsg)
        # ret = (l1 + l2) * 0.5
        # return ret.mean()
        # L2归一化
        h1_norm = F.normalize(h1, dim=1)
        h2_norm = F.normalize(h2, dim=1)

        # 计算相似度矩阵
        logits_12 = torch.mm(h1_norm, h2_norm.t())
        logits_21 = torch.mm(h2_norm, h1_norm.t())
        # 创建标签
        labels = torch.arange(h1_norm.shape[0], device=h1_norm.device)
        # 使用CrossEntropyLoss
        loss = (nn.CrossEntropyLoss()(logits_12, labels) + nn.CrossEntropyLoss()(logits_21, labels)) / 2
        return loss

    def decor(self, g):
        # decor_augmentations = [、

        #     dgl.transforms.FeatMask(self.args.decor_dfr, ['attr'])
        # ]
        decor_augmentations = [
            dgl.transforms.DropEdge(self.args.decor_der),
            dgl.transforms.FeatMask(self.args.decor_dfr, ['attr'])
        ]
        decor_sampler = dgl.dataloading.SAINTSampler('node', budget=self.args.decor_size)
        lambd = 1e-3
        g_v1 = decor_sampler.sample(g, 0).to(self.args.device)
        g_v2 = dgl_graph_copy(g_v1).to(self.args.device)
        # for aug in decor_augmentations: aug(g_v1)
        for aug in decor_augmentations: aug(g_v2)
        N = g_v1.number_of_nodes()
        h1 = self.encoder(g_v1, g_v1.ndata['attr'])
        h2 = self.encoder(g_v2, g_v2.ndata['attr'])
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)
        c = (z1 - z2) / N
        c1 = c1 / N
        c2 = c2 / N
        # 通过数据增强和去冗余约束来优化图结构数据的表征学习
        loss_inv = torch.linalg.matrix_norm(c)
        iden = torch.tensor(np.eye(c1.shape[0])).to(h1.device)
        loss_dec1 = torch.linalg.matrix_norm(iden - c1)
        loss_dec2 = torch.linalg.matrix_norm(iden - c2)

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)
        return loss

    def embed(self, g):
        if type(g.ndata['attr']) == dict:
            x = g.ndata['attr']['_N'].to(g.device)
        else:
            x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])