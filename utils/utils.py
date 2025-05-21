from datetime import datetime

import dgl
from dateutil import parser
import pytz
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import random
import torch.optim as optim

from datas import dgl_graph_copy
from utils.data_aug import find_bridges_and_articulation_points, drop_nodes, drop_edges


# Construct negative sample pairs
def calculate_overlap_matrix(graphs):
    n = len(graphs)
    overlap_matrix = np.zeros((n, n))

    # Precompute the node sets of all graphs
    node_sets = [set(g.nodes().cpu().numpy()) for g in graphs]

    # Compute the overlap matrix
    for i in range(n):
        for j in range(i + 1, n):  # Only compute the upper triangular matrix
            nodes1 = node_sets[i]
            nodes2 = node_sets[j]
            overlap = len(nodes1.intersection(nodes2))
            union = len(nodes1.union(nodes2))
            overlap_ratio = overlap / union
            overlap_matrix[i][j] = overlap_ratio
            overlap_matrix[j][i] = overlap_ratio  # The matrix is symmetric

    return overlap_matrix

def calculate_node_overlap(g1, g2):
    """Calculate node overlap between two subgraphs"""
    nodes1 = set(g1.nodes().cpu().numpy())
    nodes2 = set(g2.nodes().cpu().numpy())
    overlap = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    return overlap / union


def create_similarity_based_mask(graphs_v1, threshold=0.3):
    """Create a similarity mask based on node overlap"""
    N = len(graphs_v1)
    similarity_mask = torch.zeros((2 * N, 2 * N), dtype=bool)

    # Fill in the original positive sample pairs
    for i in range(N):
        similarity_mask[i, i + N] = True
        similarity_mask[i + N, i] = True

    # Add additional positive sample pairs based on node overlap
    for i in range(N):
        for j in range(i + 1, N):
            overlap = calculate_node_overlap(graphs_v1[i], graphs_v1[j])
            if overlap > threshold:
                # Treat subgraph pairs with high overlap as positive samples
                similarity_mask[i, j] = True
                similarity_mask[j, i] = True
                similarity_mask[i + N, j + N] = True
                similarity_mask[j + N, i + N] = True

    return similarity_mask

def sim_matrix(a, b, eps=1e-8):
    """
    Added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def create_optimizer(opt, model, lr, weight_decay):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    optimizer = None
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
    return optimizer

def print_mem(epoch):
    # Record CPU memory usage
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_mem = mem_info.rss / (1024 ** 2)  # Convert to MB
    print(f"Epoch {epoch}: CPU Memory: {cpu_mem:.2f} MB")

def random_shuffle(x, y):
    idx = list(range(len(x)))
    random.shuffle(idx)
    return x[idx], y[idx]

class DynamicTaskPriority:
    def __init__(self, num_losses, device, momentum=0.9):
        self.num_losses = num_losses
        self.momentum = momentum
        self.moving_losses = torch.ones(num_losses).to(device)
        self.initial_losses = None

    def update_weights(self, current_losses):
        # Ensure current_losses is a detached copy
        current_losses = current_losses.detach()

        if self.initial_losses is None:
            self.initial_losses = current_losses.clone()  # Use clone() instead of detach()

        # Update moving average
        self.moving_losses = (self.momentum * self.moving_losses +
                              (1 - self.momentum) * current_losses)
        # Calculate relative progress
        progress = self.moving_losses / self.initial_losses

        # Calculate weights
        weights = progress / progress.sum()
        return weights.clone()  # Return a cloned weight tensor

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    torch.backends.cudnn.enabled = False

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

def gen_train_data(g, args):
    # Create sampler and dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler([8, 8])

    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.num_nodes()).to(g.device),
        sampler,
        # batch_size=self.args.batch_size,
        batch_size=3,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    graphs_v1 = []
    # Example usage
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # blocks are the sampled subgraph list, blocks[-1] is the last layer subgraph
        block = blocks[0].to(args.device)  # 2-hop neighbor subgraph
        src, dst = block.edges()
        subgraph = dgl.graph((src, dst), num_nodes=block.num_src_nodes())
        subgraph.ndata['attr'] = block.srcdata['attr']
        subgraph.edata['attr'] = block.edata['attr']
        graphs_v1.append(subgraph)

        if len(graphs_v1) == args.batch_size: break

    if args.svd == 0:
        graphs_v2 = [dgl_graph_copy(tg) for tg in graphs_v1]
        gm_augmentations = [
            dgl.transforms.DropEdge(args.gm_edge_drop_ratio), \
            dgl.transforms.DropNode(args.gm_node_drop_ratio)
        ]
        probabilities = [0.5, 0.5]
        aug_type = np.random.choice(len(gm_augmentations),
                                    args.batch_size,
                                    replace=True,
                                    p=probabilities)
        aug_type = {k: aug_type[k] for k in range(args.batch_size)}
        for i, temp_g in enumerate(graphs_v2): gm_augmentations[aug_type[i]](temp_g)
        bg1 = dgl.batch(graphs_v1).to(args.device)
        aug_bg2 = dgl.batch(graphs_v2).to(args.device)
        aug_bg2_edge_droped = aug_bg2
    else:
        bg1 = dgl.batch(graphs_v1).to(args.device)
        be, an = find_bridges_and_articulation_points(bg1)
        if random.random() < 0.5:
            aug_bg2 = drop_nodes(bg1, an, args.gm_node_drop_ratio)
            aug_bg2_edge_droped = drop_edges(bg1, be, p=args.gm_edge_drop_ratio)
        else:
            aug_bg2 = drop_edges(bg1, be, p=args.gm_edge_drop_ratio)
            aug_bg2_edge_droped = aug_bg2

    return bg1, aug_bg2, aug_bg2_edge_droped