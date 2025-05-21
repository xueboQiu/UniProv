import copy
import os
import time

import psutil
import torch
import warnings

from eval_graph_cls import evaluate_batch_level_using_knn
from eval_graph_matching import eval_graph_matching, create_graph_matching_datasets
from eval_node_cls import evaluate_entity_level_using_knn
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata, transform_graph
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from utils.config import build_args
warnings.filterwarnings('ignore')


def eval_node_cls(main_args, model, metadata):
    model.eval()

    # print(main_args)
    dataset_name = main_args.dataset
    device = main_args.device

    malicious, _ = metadata['malicious']
    n_train = metadata['n_train']
    n_test = metadata['n_test']

    start_time = time.time()

    with torch.no_grad():
        x_train = []
        for i in range(n_train):
            g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
            x_train.append(model.embed(g).cpu().numpy())
            del g
        x_train = np.concatenate(x_train, axis=0)
        skip_benign = 0
        x_test = []
        for i in range(n_test):
            g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
            # Exclude training samples from the test set
            if i != n_test - 1:
                skip_benign += g.number_of_nodes()
            x_test.append(model.embed(g).cpu().numpy())
            del g
        x_test = np.concatenate(x_test, axis=0)

        n = x_test.shape[0]
        y_test = np.zeros(n)
        y_test[malicious] = 1.0
        malicious_dict = {}
        for i, m in enumerate(malicious):
            malicious_dict[m] = i

        # Exclude training samples from the test set
        test_idx = []
        for i in range(x_test.shape[0]):
            if i >= skip_benign or y_test[i] == 1.0:
                test_idx.append(i)
        result_x_test = x_test[test_idx]
        result_y_test = y_test[test_idx]
        del x_test, y_test
        test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test, result_y_test)
        # test_auc, test_std, _, _ = evaluate_entity_level_using_kmeans(dataset_name, x_train, result_x_test, result_y_test)

    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    end_time = time.time()

    print(f"Testing time: {end_time - start_time} seconds")
    # record memory cost
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_mem = mem_info.rss / (1024 ** 2)  # 转换为 MB
    print(f"CPU Memory: {cpu_mem:.2f} MB")
    return

def eval_graph_cls(main_args, model):
    model.eval()
    device = main_args.device
    dataset = main_args.dataset

    pooler = Pooling(main_args.pooling)
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    edge_num_list = []
    with torch.no_grad():
        for i in full:
            g = transform_graph(graphs[i][0], main_args.n_dim, main_args.e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, [2]).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
            edge_num_list.append(g.number_of_edges())
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    test_auc, test_std = evaluate_batch_level_using_knn(10, dataset, x, y)

def eval_graph_match(main_args, model):
    model.eval()
    dataset_name = main_args.dataset

    test_data_path = f'./data_cl/{dataset_name}/test_gm_pairs.pt'
    if not os.path.exists(test_data_path):
        test_dataset = create_graph_matching_datasets(main_args)
        # dump the test dataset
        torch.save(test_dataset, test_data_path)
    else:
        test_dataset = torch.load(test_data_path)

    print("Start evaluating graph matching...")
    eval_graph_matching(model, test_dataset, main_args)

def evaluate(args):
    main_args = copy.deepcopy(args)
    dataset_name = main_args.dataset
    main_args.device = device = f'cuda:{main_args.device}' if main_args.device >= 0 else "cpu"

    set_random_seed(0)
    print("evaluate the downstream task:", main_args.downstream_tasks)

    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
        # main_args.num_hidden = main_args.d
        # main_args.num_layers = main_args.l
        main_args.max_epoch = 2 if dataset_name == 'wget' else 5
        # if dataset_name == 'wget':   main_args.batch_size = 128
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
    else:
        metadata = load_metadata(dataset_name)
        n_node_feat = metadata['node_feature_dim']
        n_edge_feat = metadata['edge_feature_dim']
        main_args.num_hidden = main_args.d
        main_args.num_layers = main_args.l

    main_args.n_dim = n_node_feat
    main_args.e_dim = n_edge_feat

    # ck_name = f"./checkpoints/checkpoint-{dataset_name}_{'_'.join(main_args.tasks)}.pt"
    if args.iterate_eval_params == 0:
        if args.iterate_eval_scaling == 1:
            ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_traing_number{main_args.train_g_number}.pt")
        elif args.gnn == 'gat':
            ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_{'_'.join(main_args.tasks)}"
                       f"_epoch{main_args.max_epoch}_bu{main_args.bu}_svd{main_args.svd}_bs{main_args.batch_size}.pt")
        else:
            ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_{'_'.join(main_args.tasks)}"
                       f"_epoch{main_args.max_epoch}_bu{main_args.bu}_svd{main_args.svd}_bs{main_args.batch_size}_{main_args.gnn}.pt")
    else:
        ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_d{args.d}_l{args.l}.pt")

    model = build_model(main_args).to(device)

    if os.path.exists(ck_name):
        print("Model loaded from: ", ck_name)
        model.load_state_dict(torch.load(ck_name, map_location=main_args.device))
    else:
        print("No trained model of:", ck_name)

    if "dt_node_cls" in main_args.downstream_tasks:
        eval_node_cls(main_args, model, metadata)
    if 'dt_graph_cls' in main_args.downstream_tasks:
        eval_graph_cls(main_args, model)
    if 'dt_graph_match' in main_args.downstream_tasks:
        eval_graph_match(main_args, model)
    if 'dt_node_cluster' in main_args.downstream_tasks:
        pass


def iter_eval_params(args):
    task =  ['p_link','p_recon','p_ming','p_minn','p_misgsg','p_decor']
    # datasets = ['cadets', 'theia', 'optc_1', 'optc_2', 'optc_3', 'streamspot','trace']
    datasets = ['wget']

    device = args.device
    l = [1, 2, 3, 4, 5]
    e = [32, 64, 128, 256, 512]
    # e = []
    for embed in e:
        print("=============================================    current testing model with embedding size is ", embed)
        for dataset in datasets:
            if "wget" in dataset:
                args.dt_graph_cls = 1
                args.dt_node_cls = 0
                args.dt_graph_match = 0
                args.max_epoch=2
            elif "streamspot" in dataset:
                args.dt_graph_cls = 1
                args.dt_node_cls = 0
                args.dt_graph_match = 0
                args.max_epoch=5
            elif ('cadets' in dataset or 'theia' in dataset or 'trace' in dataset):
                args.dt_graph_cls = 0
                args.dt_node_cls = 1
                args.dt_graph_match = 1
                args.threshold = 0.9
            else:
                args.dt_graph_cls = 0
                args.dt_node_cls = 0
                args.dt_graph_match = 1
                args.threshold = 0.9

            args.downstream_tasks = [task for task in ['dt_node_cls', 'dt_graph_match', 'dt_graph_cls', 'dt_node_clt']
                                     if getattr(args, task)]
            args.dataset = dataset
            # print(f"Dataset: {dataset}, Downstream Tasks: {args.downstream_tasks}")
            args.l = 3
            args.d = embed
            args.tasks = task
            args.device = device
            evaluate(args)

    for layer in l:
        print("=============================================    current testing model with gnn layer is ", layer)
        for dataset in datasets:
            args.dataset = dataset
            if "wget" in dataset:
                args.dt_graph_cls = 1
                args.dt_node_cls = 0
                args.dt_graph_match = 0
                args.max_epoch=2
            elif "streamspot" in dataset:
                args.dt_graph_cls = 1
                args.dt_node_cls = 0
                args.dt_graph_match = 0
                args.max_epoch=5
            elif ('cadets' in dataset or 'theia' in dataset or 'trace' in dataset):
                args.dt_graph_cls = 0
                args.dt_node_cls = 1
                args.dt_graph_match = 1
                args.threshold = 0.9
            else:
                args.dt_graph_cls = 0
                args.dt_node_cls = 0
                args.dt_graph_match = 1
                args.threshold = 0.9
            args.downstream_tasks = [task for task in ['dt_node_cls', 'dt_graph_match', 'dt_graph_cls', 'dt_node_clt']
                                     if getattr(args, task)]
            args.dataset = dataset
            args.l = layer
            args.d = 64
            args.tasks = task
            args.device = device
            evaluate(args)


def iter_abl(args):

    # epochs = [50,100,150,200,250]
    # bs = [64,128,256,512,1024]

    abl_tasks = [
            ['p_link','p_recon','p_ming','p_minn','p_misgsg','p_decor'],    # full
            ['p_link','p_recon','p_ming','p_minn','p_misgsg'],   # lack of decor
            ['p_link','p_recon','p_ming','p_misgsg','p_decor'],   # lack of minn
            ['p_link','p_recon','p_ming','p_minn','p_decor'],    # lack of misgsg
            ['p_link','p_recon','p_minn','p_misgsg','p_decor'],      # lack of ming
            ['p_link','p_ming','p_minn','p_misgsg','p_decor'],    # lack of recon
            ['p_recon','p_ming','p_minn','p_misgsg','p_decor'],     # lack of link
        ]
    # dt_tasks = ['dt_node_cls', 'dt_graph_cls', 'dt_graph_match']
    dt_tasks = ['dt_graph_cls']
    device = args.device
    args.max_epoch = 50
    args.batch_size = 512
    print("# =======================# ======================= Testing abl_tasks effect")
    for tasks in abl_tasks:
        print("current task: ", tasks)
        for dt in dt_tasks:
            args.tasks = tasks
            if "node_cls" in dt:
                args.downstream_tasks = ['dt_node_cls']
                args.dataset = 'cadets'
                args.device = device
                evaluate(args)
            elif "graph_cls" in dt:
                args.downstream_tasks = ['dt_graph_cls']
                args.dataset = 'wget'
                args.device = device
                evaluate(args)
            else:
                args.downstream_tasks = ['dt_graph_match']
                for dataset in ['optc_1', 'optc_2', 'optc_3']:
                    args.dataset = dataset
                    args.device = device
                    evaluate(args)
            # args.bu = 1
            # args.svd = 0
    print("# =======================# ======================= Testing bu effect")
    for dt in dt_tasks:
        args.bu = 0
        args.tasks = abl_tasks[0]
        if "node_cls" in dt:
            args.downstream_tasks = ['dt_node_cls']
            args.dataset = 'cadets'
            args.device = device
            evaluate(args)
        elif "graph_cls" in dt:
            args.downstream_tasks = ['dt_graph_cls']
            args.dataset = 'wget'
            args.device = device
            evaluate(args)
        else:
            args.downstream_tasks = ['dt_graph_match']
            for dataset in ['optc_1', 'optc_2', 'optc_3']:
                args.dataset = dataset
                args.device = device
                evaluate(args)
    print("# =======================# ======================= Testing gnn backbones effect")
    args.bu = 1
    args.tasks = abl_tasks[0]
    backbones = ['gnn', 'gin', 'gcn', 'graphsage']
    for backbone in backbones:
        for dt in dt_tasks:
            print("current gnn backbone is ", backbone)
            args.gnn = backbone
            args.tasks = abl_tasks[0]
            if "node_cls" in dt:
                args.downstream_tasks = ['dt_node_cls']
                args.dataset = 'cadets'
                args.device = device
                evaluate(args)
            elif "graph_cls" in dt:
                args.downstream_tasks = ['dt_graph_cls']
                args.dataset = 'wget'
                args.device = device
                evaluate(args)
            else:
                args.downstream_tasks = ['dt_graph_match']
                for dataset in ['optc_1', 'optc_2', 'optc_3']:
                    args.dataset = dataset
                    args.device = device
                    evaluate(args)


def iter_eval_scaling(args):
    for train_g in [1, 2, 3, 4]:
        print("========================================= train_g number: ", train_g)
        for dataset in ['cadets', 'theia', 'trace']:
            args.d = 64
            args.l = 3
            args.dataset = dataset
            args.downstream_tasks = ['dt_node_cls']
            args.train_g_number = train_g
            evaluate(args)

def iter_eval(args):
    if args.iterate_eval_params == 1:
        iter_eval_params(args)
        
    if args.iterate_abl == 1:
        iter_abl(args)
    
    if args.iterate_eval_scaling == 1:
        iter_eval_scaling(args)

if __name__ == '__main__':
    args = build_args()

    if args.iterate_eval == 0:
        evaluate(args)
    else:
        iter_eval(args)