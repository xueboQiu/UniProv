import json
import os
import pickle
import random
import time

import numpy as np
import torch.nn.functional as F  # 这里导入了F
import dgl
import networkx as nx
import torch
from networkx.readwrite import json_graph
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from tqdm import tqdm

from model.autoencoder import build_model
from utils.loaddata import transform_graph


def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)
def fea_init(dataset, g):

    with open(f"./data_cl/{dataset}/" + 'edge_type_dict.json', 'r') as f:
        edge_type_dict = json.load(f)

    with open(f"./data_cl/{dataset}/" + 'node_type_dict.json', 'r') as f:
        node_type_dict = json.load(f)

    edge_type_cnt = len(edge_type_dict)
    # convert edge features to multi-hot encoding
    for (u, v, k) in g.edges(keys=True):
        edge_types = g[u][v][k]['edge_type']
        # multiu_hot_encodings = [0] * (edge_type_cnt)
        # for et in edge_types:
        #     if et in edge_type_dict:
        #         multiu_hot_encodings[edge_type_dict[et]] = 1
        # g[u][v][k]['type'] = torch.tensor(multiu_hot_encodings)
        g[u][v][k]['type'] = torch.tensor(edge_type_dict[edge_types[0]])

    # assign abstract node types
    for u in g.nodes():
        node_type = g.nodes[u]['type']
        type_idx = -1
        for k, v in node_type_dict.items():
            if node_type.lower() in k.lower():
                type_idx = v
                break
        if type_idx == -1:
            print("Unknown node type: ", node_type)
            exit(1)
        g.nodes[u]['type'] = torch.tensor(type_idx)

    return g
def is_duplicate(gt1, gt2, thre):
    gt1 = gt1[0]
    gt2 = gt2[0]

    uuids1 = set([gt1.nodes[u]['uuid'] for u in gt1.nodes()])
    uuids2 = set([gt2.nodes[u]['uuid'] for u in gt2.nodes()])
    simi = len(uuids1.intersection(uuids2)) / min(len(uuids1), len(uuids2))
    if simi > thre:
        return True
    return False
def deduplicate(gts):

    # 逐个去重
    dedup_gts = []
    dedup_idxes = {}
    for i, gt in enumerate(gts):
        is_dup = False
        for j,dedup_gt in enumerate(dedup_gts):
            if is_duplicate(gt, dedup_gt,0.9):
                is_dup = True
                dedup_idxes[i] = j
                print(f"duplicate:{gt[1]},{dedup_gt[1]}")
                break
        if not is_dup:
            dedup_idxes[i] = len(dedup_gts)
            dedup_gts.append(gt)
        # else:
        #     if len(gt[0].nodes) > len(dedup_gt[0].nodes):
        #         dedup_gts.remove(dedup_gt)
        #         dedup_gts.append(gt)

    print("deduplicate number of gts:", len(dedup_gts))
    return dedup_gts, dedup_idxes
def load_and_initialize_graphs(file_paths, dataset):
    graphs = []
    for file_path in file_paths:
        graph = read_json_graph(file_path)
        graph = fea_init(dataset, graph)
        graphs.append((graph, os.path.basename(file_path)))
    return graphs

def load_pois(dataset):
    in_path = f"../GMPT/dataset/darpa_{dataset}/pois"
    files = os.listdir(in_path)

    pois_uuids = []
    for file in files:
        with open(f"{in_path}/{file}", 'r') as f:
            for line in f:
                pois_uuids.append(line.strip().split(":")[0])

    return pois_uuids

def graph_reduction(g):
    # 将相同名字且类型相同的节点合并
    # 解除冻结图
    sbg = g.copy()
    nname2id = {sbg.nodes[node]["type"] + sbg.nodes[node]["name"]: node for node in sbg.nodes}

    # print("nname2id:",nname2id)
    nodes = list(sbg.nodes)
    for node in nodes:
        node_name = sbg.nodes[node]["type"] + sbg.nodes[node]["name"]
        if node_name.endswith("/"):
            sbg.remove_node(node)
            continue
        if node == nname2id[node_name]:
            continue
        # 得到当前节点的所有边，并转移至节点nname2id[node_name]中
        for edge in list(sbg.in_edges(node, keys=True)) + list(sbg.out_edges(node, keys=True)):
            src, dst, key = edge
            if src == node:
                src, dst = dst, src
            sbg.add_edge(src, nname2id[node_name], key=key, **sbg.edges[edge])
        sbg.remove_node(node)

    # 冻结图
    sbg = nx.freeze(sbg)

    return sbg


def load_poi_names(gts):

    names = []
    for k,v in gts:
        for node in k.nodes:
            if 'remote_ip' in k.nodes[node]:
                names.append(k.nodes[node]['remote_ip'])
            elif 'image_path' in k.nodes[node]:
                names.append(k.nodes[node]['image_path'])
            elif 'file_path' in k.nodes[node]:
                names.append(k.nodes[node]['file_path'])

    return list(set(names))

def create_graph_matching_datasets(args):
    dataset = args.dataset
    if "optc" not in dataset:
        gt_filepaths = [os.path.join(f'../GMPT/dataset/darpa_{dataset}/groundtruth/', file) for file in
                        os.listdir(f'../GMPT/dataset/darpa_{dataset}/groundtruth/')]
        sg_filepaths = [os.path.join(f'../GMPT/dataset/darpa_{dataset}/sampled_graphs/', file) for file in
                        os.listdir(f'../GMPT/dataset/darpa_{dataset}/sampled_graphs/')]
    else:
        sce_maps = {"optc_1":"0201", "optc_2":"0501", "optc_3":"0051"}
        gt_filepaths = [os.path.join(f'../GMPT/dataset/darpa_optc/groundtruth/', file) for file in
                        os.listdir(f'../GMPT/dataset/darpa_optc/groundtruth/') if sce_maps[dataset] in file]
        sg_filepaths = [os.path.join(f'../GMPT/dataset/darpa_optc/sampled_graphs/', file) for file in
                        os.listdir(f'../GMPT/dataset/darpa_optc/sampled_graphs/') if sce_maps[dataset] in file]
    # 加载和初始化图数据
    gts = load_and_initialize_graphs(gt_filepaths,dataset)
    sgs = load_and_initialize_graphs(sg_filepaths,dataset)
    # 移除节点过多的gt graphs
    gts = [gt for gt in gts if len(gt[0].nodes) < 50]
    sgs = [sg for sg in sgs if len(sg[0].nodes) < 60]

    # 移除相似的gt graphs
    dedu_gts, dedup_idxes = deduplicate(gts)
    query_samples = [gt[0] for gt in dedu_gts]

    dg_query_samples = []
    for g in query_samples:
        dg_query_samples.append(transform_graph(convert_multidigraph_to_dgl(g), args.n_dim, args.e_dim))
    # 使用列表推导式预生成所有可能的查询索引
    all_query_indices = set(range(len(query_samples)))
    pos_samples = []
    neg_samples = []
    for sg in sgs:
        query_idx = -1
        for i, gt in enumerate(dedu_gts):
            if is_duplicate(gt, sg, 0.7):
                query_idx = i
                # print(f"==============gt file:{gt[1]}, gt num:{len(gt[0].nodes)}. sg file:{sg[1]}, sg num:{len(sg[0].nodes)}, query_idx:{query_idx}")
                break
        if query_idx == -1:
            print("missing query_idx")
            continue
        dg_sg = transform_graph(convert_multidigraph_to_dgl(sg[0]), args.n_dim, args.e_dim)

        pos_samples.append((dg_sg, dg_query_samples[query_idx]))
        # 添加负样本 - 使用集合操作
        neg_indices = all_query_indices - {query_idx}
        neg_samples.extend((dg_sg, dg_query_samples[i]) for i in neg_indices)
        # print(f"==============gt file:{dedu_gts[dedup_idxes[query_idx]][1]}, gt num:{len(dedu_gts[dedup_idxes[query_idx]][0].nodes)}. sg file:{sg[1]}, sg num:{len(sg[0].nodes)}, query_idx:{query_idx}")

    # 加载negative test samples
    train_gs = [dgl.from_networkx(
        nx.node_link_graph(g[0]),
        node_attrs=['type'],
        edge_attrs=['type']
    ) for g in pickle.load(open(f'./data_cl/{dataset}/train.pkl', 'rb'))]
    train_g = dgl.batch(train_gs)
    # 采样1000个两层的邻居的负样本子图
    for _ in tqdm(range(500)):
        cen_id = random.randint(0, train_g.number_of_nodes() - 1)
        def sample_neighbors(graph, nodes, fanout):
            try:
                # 确保节点ID在有效范围内
                valid_nodes = nodes[nodes < graph.number_of_nodes()]
                if len(valid_nodes) == 0:
                    return torch.tensor([], dtype=torch.long)
                # 获取出边邻居
                _, out_neighs = graph.out_edges(valid_nodes)
                # 获取入边邻居
                in_neighs, _ = graph.in_edges(valid_nodes)
                # 合并所有邻居并去重
                neighs = torch.cat([out_neighs, in_neighs]).unique()
                # 确保采样的邻居节点ID也是有效的
                neighs = neighs[neighs < graph.number_of_nodes()]
                # 如果邻居数量大于fanout，随机采样
                if len(neighs) > fanout:
                    perm = torch.randperm(len(neighs))
                    return neighs[perm[:fanout]]
                return neighs
            except Exception as e:
                print(e)
                # 如果出现错误，返回空张量
                return torch.tensor([], dtype=torch.long)
        # 第一层采样
        fanout1 = fanout2 = random.randint(8, 13)
        first_nodes = sample_neighbors(train_g, torch.tensor([cen_id]), fanout1)
        # 检查第一层采样结果
        if len(first_nodes) == 0:
            continue  # 如果没有有效邻居，跳过这次采样
        if len(first_nodes) < fanout1:
            fanout2 = fanout2 + fanout1 - len(first_nodes)
        # 第二层采样
        second_nodes = sample_neighbors(train_g, first_nodes, fanout2)
        # 检查是否有有效的采样结果
        if len(second_nodes) == 0:  continue
        # 合并所有采样的节点
        all_nodes = torch.unique(torch.cat([
            torch.tensor([cen_id]),
            first_nodes,
            second_nodes
        ]))
        # 构建负样本子图
        dg_bg = transform_graph(dgl.node_subgraph(train_g, all_nodes), args.n_dim, args.e_dim)
        neg_samples.extend((dg_bg, dg_query_samples[i]) for i in all_query_indices)

    return (pos_samples, neg_samples)

def convert_multidigraph_to_dgl(multi_g):
    """
    将 NetworkX MultiDiGraph 转换为 DGLGraph
    方案1: 保留所有平行边，使用边的索引区分不同的边
    """

    # 创建节点ID映射
    unique_nodes = sorted(list(multi_g.nodes()))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    reverse_mapping = {new_id: old_id for old_id, new_id in node_mapping.items()}

    # 收集所有边及其属性
    edges_src = []
    edges_dst = []
    edge_attrs = []
    # 遍历所有边
    for u, v, key, data in multi_g.edges(data=True, keys=True):
        edges_src.append(node_mapping[u])
        edges_dst.append(node_mapping[v])
        if data:
            edge_attrs.append(data)

    # 创建DGLGraph
    g = dgl.graph((edges_src, edges_dst), num_nodes=len(node_mapping))

    # 添加边属性
    if edge_attrs:
        g.edata['type'] = torch.stack([attr['type'] for attr in edge_attrs])
    # 添加节点属性
    node_attrs = [multi_g.nodes[old_id]['type']
                 for new_id, old_id in reverse_mapping.items()]
    g.ndata['type'] = torch.stack(node_attrs)

    return g
def eval_graph_matching(model, test_dataset, args):

    device = args.device
    threshold = args.threshold

    pos_samples, neg_samples = test_dataset

    def process_graph_pairs(model, samples, device):
        # 批量处理图对
        g1_batch = [g1.to(device) for g1, _ in samples]
        g2_batch = [g2.to(device) for _, g2 in samples]

        # 批量获取嵌入
        with torch.no_grad():  # 如果是在评估模式下，添加这个可以节省内存
            emb1_batch = torch.stack([torch.mean(model.embed(g1), dim=0) for g1 in g1_batch])
            emb2_batch = torch.stack([torch.mean(model.embed(g2), dim=0) for g2 in g2_batch])

        # 批量计算相似度
        similarities = F.cosine_similarity(emb1_batch, emb2_batch)
        return similarities.cpu()

    # 主要处理流程
    pos_scores = process_graph_pairs(model, pos_samples, device)
    neg_scores = process_graph_pairs(model, neg_samples, device)

    # 将相似度值转换为预测标签
    pred_pos = (pos_scores >= threshold).float()
    pred_neg = (neg_scores >= threshold).float()

    print("avg of pos_scores:", torch.mean(pos_scores).item())
    # 找到负样本中的前五最大值的索引
    _, topk_indices = torch.topk(neg_scores, 5)
    print("top 5 neg scores:", neg_scores[topk_indices])
    print("avg of neg_scores:", torch.mean(neg_scores).item())

    # 准备真实标签和预测分数
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores.numpy(), neg_scores.numpy()])
    # 计算 AUC
    auc_score = roc_auc_score(y_true, y_scores)

    # 计算评估指标
    tp = pred_pos.sum().item()
    fn = (1 - pred_pos).sum().item()
    fp = pred_neg.sum().item()
    tn = (1 - pred_neg).sum().item()
    # 计算综合指标
    total = len(pos_scores) + len(neg_scores)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("Evaluation Results:")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # 显示负样本中的前五最大值的图
    # for i in topk_indices:
    #     subgraph_abstract_vision(neg_samples[i], args.dataset)

def subgraph_abstract_vision(test_dataset, dataset):
    def show(vis_graph, node_type_dict, edge_type_dict):

        nx_vis_graph = dgl.to_networkx(vis_graph,
                               node_attrs=['type'],  # 列出需要转换的节点特征
                              edge_attrs=['type'])

        # sim_node_type_dict = {}
        # for node_type, type_id in node_type_dict.items():
        #     # 转换节点类型
        #     if "PROCESS" in node_type.upper():
        #         sim_node_type_dict["PROCESS"] = type_id
        #     elif "FILE" in node_type.upper():
        #         sim_node_type_dict["FILE"] = type_id
        #     elif "FLOW" in node_type.upper():
        #         sim_node_type_dict["FLOW"] = type_id

        node_type_dict_reverse = {v: k for k, v in node_type_dict.items()}
        from matplotlib import pyplot as plt
        # pyg transform to nxg transform
        nodes_labels = {}
        for node_id, node_attr in list(nx_vis_graph.nodes(data=True)):
            nodes_labels[node_id] = node_type_dict_reverse[node_attr["type"].item()]

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(nx_vis_graph, k=1.5)
        # 定义每种类型的节点样式和颜色
        shapes = {
            'PROCESS': 's',  # square
            'FILE': 'o',  # circle
            'FLOW': 'd',  # diamond
        }
        colors = {
            'PROCESS': '#C3C5C7',
            'FILE': '#D19494',
            'FLOW': '#8FC3E4',
        }
        # 绘制每种类型的节点
        for node_type, shape in shapes.items():
            nodes = [n for n in nx_vis_graph.nodes if nx_vis_graph.nodes[n]['type'] == node_type]
            poi_nodes = [n for n in nodes if 'poi' in nx_vis_graph.nodes[n]]
            normal_nodes = [n for n in nodes if 'poi' not in nx_vis_graph.nodes[n]]
            # 绘制没有 poi 属性的节点
            if normal_nodes:
                nx.draw_networkx_nodes(
                    nx_vis_graph, pos,
                    nodelist=normal_nodes,
                    node_color=colors[node_type],
                    node_shape=shape,
                    node_size=1200,
                    alpha=0.9
                )
            # 绘制具有 poi 属性的节点
            if poi_nodes:
                nx.draw_networkx_nodes(
                    nx_vis_graph, pos,
                    nodelist=poi_nodes,
                    node_color='red',  # 红色
                    node_shape=shape,
                    node_size=1200,
                    alpha=0.9
                )
        # 绘制边
        nx.draw_networkx_edges(nx_vis_graph, pos, arrowstyle='-|>', arrowsize=8, edge_color="grey")
        nx.draw_networkx_edge_labels(nx_vis_graph, pos,
                                     edge_labels={(e1, e2): edge_attr['type'] for e1, e2, edge_attr in
                                                  list(nx_vis_graph.edges(data=True))},
                                     font_color='red',
                                     font_size=5,
                                     )
        nx.draw_networkx_labels(nx_vis_graph, pos, labels=nodes_labels, font_size=10, font_color='black',
                                font_weight='bold')

        plt.axis('off')
        plt.show()

    with open(f"./data_cl/{dataset}/node_type_dict.json", 'r') as f:
        node_type_dict = json.load(f)
    with open(f"./data_cl/{dataset}/edge_type_dict.json", 'r') as f:
        edge_type_dict = json.load(f)
    query_sample, sampled_sample = test_dataset

    show(query_sample, node_type_dict, edge_type_dict)
    show(sampled_sample, node_type_dict, edge_type_dict)
