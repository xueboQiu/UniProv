import random

import dgl
import numpy as np
import torch
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

from datas import dgl_graph_copy


class CustomDropNode:
    def __init__(self, p, keep_nodes=None):
        """
        Custom node drop transformation

        Parameters:
        - p: Drop probability (between 0 and 1)
        - keep_nodes: List of node IDs to keep
        """
        self.p = p
        self.keep_nodes = set(keep_nodes) if keep_nodes is not None else set()

    def __call__(self, g):
        if self.p <= 0:
            return g

        # Get all node IDs
        all_nodes = set(range(g.number_of_nodes()))

        # Get droppable nodes (excluding nodes to keep)
        droppable_nodes = list(all_nodes - self.keep_nodes)

        if not droppable_nodes:  # If no nodes can be dropped, return the original graph
            return g

        # Calculate the number of nodes to drop
        num_drop = int(len(droppable_nodes) * self.p)

        # Randomly select nodes to drop
        drop_nodes = np.random.choice(
            droppable_nodes,
            size=num_drop,
            replace=False
        )

        # Create a new graph (remove selected nodes)
        return g.remove_nodes(drop_nodes)

def drop_nodes(bg, an, p=0.2):

    g = dgl_graph_copy(bg)
    # Get all node IDs
    all_nodes = set(range(g.number_of_nodes()))
    # Get droppable nodes (excluding nodes to keep)
    droppable_nodes = list(all_nodes - set(an))

    if not droppable_nodes:  # If no nodes can be dropped, return the original graph
        return g
    # Calculate the number of nodes to drop
    num_drop = int(len(droppable_nodes) * p)

    # Randomly select nodes to drop
    drop_nodes = np.random.choice(
        droppable_nodes,
        size=num_drop,
        replace=False
    )
    # Collect all edges to be deleted
    src_nodes = []
    dst_nodes = []

    # Process selected nodes
    for node in drop_nodes:
        # Get all edges related to the node
        in_edges = g.in_edges([node])
        out_edges = g.out_edges([node])

        # Collect incoming edges
        src_nodes.extend(in_edges[0].tolist())
        dst_nodes.extend(in_edges[1].tolist())

        # Collect outgoing edges
        src_nodes.extend(out_edges[0].tolist())
        dst_nodes.extend(out_edges[1].tolist())

        # Set the node's 'attr' attribute to 0
        if 'attr' in g.ndata:
            g.ndata['attr'][node] = 0

    # If there are edges to delete
    if src_nodes:
        # Create tensors and move to the correct device
        src_tensor = torch.tensor(src_nodes, device=g.device)
        dst_tensor = torch.tensor(dst_nodes, device=g.device)

        # Remove all collected edges
        g.remove_edges(g.edge_ids(src_tensor, dst_tensor))

    return g

def drop_edges(bg, be, p=0.2):
    """
    Drop edges but keep bridge edges

    Parameters:
    - g: DGL graph
    - bridge_edges: List of bridge edges, a tensor of shape (N, 2)
    - p: Drop probability
    """
    # Clone the graph to avoid modifying the original
    g = dgl_graph_copy(bg)

    # Get all edges
    all_edges = set(zip(g.edges()[0].tolist(), g.edges()[1].tolist()))

    # Convert bridge edges to a set (consider both directions since it's an undirected graph)
    bridge_set = set()
    for edge in be:
        u, v = edge[0], edge[1]
        bridge_set.add((u, v))
        bridge_set.add((v, u))

    # Get droppable edges (excluding bridge edges)
    droppable_edges = list(all_edges - bridge_set)

    if not droppable_edges:  # If no edges can be dropped, return the original graph
        return g

    # Calculate the number of edges to drop
    num_drop = int(len(droppable_edges) * p)

    # Randomly select edges to drop
    drop_idx = np.random.choice(
        len(droppable_edges),
        size=num_drop,
        replace=False
    )
    edges_to_drop = [droppable_edges[i] for i in drop_idx]

    # Convert edges to be dropped to lists of source and destination nodes
    src_nodes = []
    dst_nodes = []
    for edge in edges_to_drop:
        src_nodes.append(edge[0])
        dst_nodes.append(edge[1])

    # Create tensors and move to the correct device
    src_tensor = torch.tensor(src_nodes, device=g.device)
    dst_tensor = torch.tensor(dst_nodes, device=g.device)

    # Remove selected edges
    g.remove_edges(g.edge_ids(src_tensor, dst_tensor))

    return g

def find_articulation_points_dgl(g):
    def dfs(node, parent):
        nonlocal time
        visited[node] = 1
        disc[node] = time  # Initialize discovery time
        low[node] = time  # Initialize lowest reachable time
        time += 1
        children = 0  # Child count

        # Get all neighbors of the current node
        neighbors = g.successors(node)

        for neighbor in neighbors:
            neighbor = neighbor.item()
            if neighbor == parent:  # Ignore edge back to parent
                continue
            if visited[neighbor] == 0:
                children += 1
                dfs(neighbor, node)

                # Update the low value of the current node
                low[node] = min(low[node], low[neighbor])

                # Check articulation point conditions
                if parent is None and children > 1:  # Root node case
                    articulation_points[node] = 1
                if parent is not None and low[neighbor] >= disc[node]:  # Non-root node case
                    articulation_points[node] = 1
            else:
                # Update low value during backtracking
                low[node] = min(low[node], disc[neighbor])

    num_nodes = g.num_nodes()

    # Initialize tensors
    visited = torch.zeros(num_nodes)
    disc = torch.zeros(num_nodes)
    low = torch.zeros(num_nodes)
    articulation_points = torch.zeros(num_nodes)
    time = 0

    # Perform DFS for each node
    for node in range(num_nodes):
        if visited[node] == 0:
            dfs(node, None)

    return torch.nonzero(articulation_points).squeeze()

def find_bridge_edges_dgl(g):
    def dfs(node, parent):
        nonlocal time
        visited[node] = 1
        disc[node] = time  # Initialize discovery time
        low[node] = time  # Initialize lowest reachable time
        time += 1

        # Get all neighbors of the current node
        neighbors = g.successors(node)

        for neighbor in neighbors:
            neighbor = neighbor.item()
            if neighbor == parent:  # Ignore edge back to parent
                continue

            if visited[neighbor] == 0:
                dfs(neighbor, node)
                # Update the low value of the current node
                low[node] = min(low[node], low[neighbor])

                # Check bridge edge condition: if the lowest reachable time of the child node is greater than the discovery time of the current node, the edge is a bridge
                if low[neighbor] > disc[node]:
                    bridge_edges.append((node, neighbor))
            else:
                # Update low value during backtracking
                low[node] = min(low[node], disc[neighbor])

    num_nodes = g.num_nodes()

    # Initialize tensors
    visited = torch.zeros(num_nodes)
    disc = torch.zeros(num_nodes)
    low = torch.zeros(num_nodes)
    time = 0
    bridge_edges = []  # Store bridge edges

    # Perform DFS for each node
    for node in range(num_nodes):
        if visited[node] == 0:
            dfs(node, None)

    # Convert result to tensor
    if bridge_edges:
        return torch.tensor(bridge_edges)
    else:
        return torch.zeros((0, 2), dtype=torch.long)  # Return empty tensor if no bridge edges


def gen_cl_data(g, args):
    # Create sampler and dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])

    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.num_nodes()).to(g.device),
        sampler,
        # batch_size=self.args.batch_size,
        batch_size=5,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    graphs_v1 = []
    # Example usage
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # blocks is a list of sampled subgraphs, blocks[-1] is the subgraph of the last layer
        block = blocks[0].to(args.device)  # 2-hop neighbor subgraph
        src, dst = block.edges()
        subgraph = dgl.graph((src, dst), num_nodes=block.num_src_nodes())
        subgraph.ndata['attr'] = block.srcdata['attr']
        subgraph.edata['attr'] = block.edata['attr']
        graphs_v1.append(subgraph.to('cpu'))

        if len(graphs_v1) == args.batch_size: break

    graphs_v2 = [dgl_graph_copy(g) for g in graphs_v1]

    bg1 = dgl.batch(graphs_v1).to(args.device)
    bg2 = dgl.batch(graphs_v2).to(args.device)

    if args.svd == 1:
        # be, an = parallel_custom_aug(graphs_v2, batch_size=args.batch_size//16, n_jobs=-1)
        be, an = find_bridges_and_articulation_points(bg1)
    else:
        be, an  = [], []
    if random.random() < 0.5:
        aug_bg2 = drop_nodes(bg2, an, args.gm_node_drop_ratio)
        aug_bg2_edge_droped = drop_edges(bg2, be, args.gm_node_drop_ratio)
    else:
        aug_bg2 = drop_edges(bg2, be, args.gm_node_drop_ratio)
        aug_bg2_edge_droped = aug_bg2

    return bg1, aug_bg2, aug_bg2_edge_droped

def process_single_graph_custom(graph):
    """Process a single graph and return articulation points"""
    # Ensure the graph is on CPU
    if isinstance(graph, dgl.DGLGraph):
        graph = graph.to('cpu')
    # Place your articulation point finding logic here
    be, an = find_bridges_and_articulation_points(graph)
    return be, an

def process_single_graph_random(graph):
    """Process a single graph and return articulation points"""
    # Ensure the graph is on CPU
    if isinstance(graph, dgl.DGLGraph):
        graph = graph.to('cpu')
    if random.random() < 0.5:
        aug_bg2 = drop_nodes(graph, [], 0.3)
        aug_bg2_edge_droped = drop_edges(graph, [], 0.3)
    else:
        aug_bg2 = drop_edges(graph, [], 0.3)
        aug_bg2_edge_droped = aug_bg2
    return aug_bg2

def parallel_random_aug(graph_list, batch_size=1000, n_jobs=-1):
    """
        Parallel processing of articulation point finding for multiple graphs

        Parameters:
        - graph_list: List of graphs
        - batch_size: Number of graphs processed per batch
        - n_jobs: Number of CPU cores to use, -1 for all cores
        """
    # Split the graph list into multiple batches
    num_graphs = len(graph_list)
    num_batches = (num_graphs + batch_size - 1) // batch_size

    be, an = [], []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_graphs)
        batch_graphs = graph_list[start_idx:end_idx]
        # Parallel processing of the current batch
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_graph_random)(g) for g in batch_graphs
        )
        for be_, an_ in batch_results:
            be.append(be_)
            an.append(an_)

    return be, an

def parallel_custom_aug(graph_list, batch_size=1000, n_jobs=-1):
    """
    Parallel processing of articulation point finding for multiple graphs

    Parameters:
    - graph_list: List of graphs
    - batch_size: Number of graphs processed per batch
    - n_jobs: Number of CPU cores to use, -1 for all cores
    """
    # Split the graph list into multiple batches
    num_graphs = len(graph_list)
    num_batches = (num_graphs + batch_size - 1) // batch_size

    be, an = [], []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_graphs)
        batch_graphs = graph_list[start_idx:end_idx]

        # Parallel processing of the current batch
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_graph_custom)(g) for g in batch_graphs
        )
        for be_, an_ in batch_results:
            be.extend(be_)
            an.extend(an_)

    return be, an

def find_bridges_and_articulation_points(g):
    def dfs(node, parent):
        nonlocal time
        visited[node] = 1
        disc[node] = time
        low[node] = time
        time += 1

        children = 0

        neighbors = g.successors(node)
        for neighbor in neighbors:
            neighbor = neighbor.item()
            if neighbor == parent:
                continue

            if visited[neighbor] == 0:
                children += 1
                dfs(neighbor, node)

                low[node] = min(low[node], low[neighbor])

                if parent is not None and low[neighbor] >= disc[node]:
                    articulation_points.add(node)

                if low[neighbor] > disc[node]:
                    bridge_edges.append((node, neighbor))
            else:
                low[node] = min(low[node], disc[neighbor])

        if parent is None and children > 1:
            articulation_points.add(node)

    num_nodes = g.num_nodes()
    visited = torch.zeros(num_nodes)
    disc = torch.zeros(num_nodes)
    low = torch.zeros(num_nodes)
    time = 0

    bridge_edges = []
    articulation_points = set()

    for node in range(num_nodes):
        if visited[node] == 0:
            dfs(node, None)

    return bridge_edges, list(articulation_points)

def create_example_dgl_graph():

    edges_src = [0, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 7, 8]
    edges_dst = [1, 0, 2, 3, 4, 1, 5, 1, 1, 5, 2, 4, 6, 7, 5, 5, 5]

    g = dgl.graph((edges_src, edges_dst))
    return g


if __name__ == "__main__":
    g = create_example_dgl_graph()

    art_points = find_articulation_points_dgl(g)
    bri_edges = find_bridge_edges_dgl(g)

    print("Discovered bridge edges:",  bri_edges.tolist())
    print("Articulation points:", art_points.tolist())

    ab, be = find_bridges_and_articulation_points(g)
    print(ab, be)