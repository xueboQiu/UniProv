import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os

from gradate.utils import feat_init


class StreamspotDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(StreamspotDataset, self).__init__(name=name)
        if name == 'streamspot':
            path = './data_cl/streamspot'
            num_graphs = 600
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx + 1))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 300 <= idx <= 399:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class WgetDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(WgetDataset, self).__init__(name=name)
        if name == 'wget':
            path = './data_cl/wget/final'
            num_graphs = 150
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 0 <= idx <= 24:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def load_rawdata(name):
    if name == 'streamspot':
        path = './data/streamspot'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:
            raw_data = StreamspotDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    elif name == 'wget':
        path = './data_cl/wget'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:
            raw_data = WgetDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    else:
        raise NotImplementedError
    return raw_data


def load_batch_level_dataset(dataset_name):
    dataset = load_rawdata(dataset_name)
    graph, _ = dataset[0]
    node_feature_dim = 0
    for g, _ in dataset:
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item())
    edge_feature_dim = 0
    for g, _ in dataset:
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item())
    node_feature_dim += 1
    edge_feature_dim += 1
    full_dataset = [i for i in range(len(dataset))]
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim))

    return {'dataset': dataset,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            'e_feat': edge_feature_dim}


def transform_graph(g, node_feature_dim, edge_feature_dim):

    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def preload_entity_level_dataset(dataset):
    path = './data_cl/' + dataset
    # if os.path.exists(path + '/metadata.json'):
    if os.path.exists(path + '/metadata.json'):
        pass
    else:

        # feat_init(train_gs)
        print('transforming')
        # train_gs = pkl.load(open(path + '/train.pkl', 'rb'))
        # test_gs = pkl.load(open(path + '/test.pkl', 'rb'))

        # train_gs_ids = {i: {node['id']: node['uuid'] for node in g['nodes']} for i, g in enumerate(train_gs)}
        # test_gs_ids = {i: {node['id']: node['uuid'] for node in g['nodes']} for i, g in enumerate(test_gs)}
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))
        train_gs = [dgl.from_networkx(
            nx.node_link_graph(g[0]),
            node_attrs=['type'],
            edge_attrs=['type']
            # edge_attrs=['type','edge_time']
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))]
        print('transforming')
        test_gs = [dgl.from_networkx(
            nx.node_link_graph(g[0]),
            # node_attrs=['type', 'label'],
            node_attrs=['type'],
            edge_attrs=['type']
            # edge_attrs=['type','edge_time']
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))]

        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for i, g in enumerate(test_gs):
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            # g = feat_init(g, names, test_gs_ids[i])
            result_test_gs.append(g)
        result_train_gs = []
        for i, g in enumerate(train_gs):
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            # g = feat_init(g, names, train_gs_ids[i])
            result_train_gs.append(g)

        # node_feature_dim += 16  # name embedding
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        print("size of the malicious nodes: ", len(malicious))
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + '/train{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + '/test{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)

def load_metadata(path):
    preload_entity_level_dataset(path)
    with open('./data_cl/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    preload_entity_level_dataset(path)
    with open('./data_cl/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data
