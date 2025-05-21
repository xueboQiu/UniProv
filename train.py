import copy
import os
import random
import time
from itertools import combinations

import numpy as np
import psutil
import torch.nn.functional as F
import dgl
import torch
import warnings

from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datas import Graph_Dataset, build_collate
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata, transform_graph
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from dgl.dataloading import GraphDataLoader, MultiLayerNeighborSampler
from utils.utils import set_random_seed, create_optimizer, print_mem, DynamicTaskPriority
from utils.config import build_args
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def get_all_task_combinations(base_tasks=None):
    optional_tasks = ['p_ming', 'p_minn', 'p_misgsg', 'p_decor']
    base_tasks = base_tasks or ['p_link', 'p_recon']

    all_combinations = []
    # 生成1到4个可选任务的所有组合
    for r in range(1, len(optional_tasks) + 1):
        combinations_r = list(combinations(optional_tasks, r))
        # 将基础任务添加到每个组合中
        combinations_with_base = [base_tasks + list(combo) for combo in combinations_r]
        all_combinations.extend(combinations_with_base)

    return all_combinations


def all_iter(args):
    # datasets = ['cadets', 'trace', 'theia','optc_1','optc_2','optc_3', 'wget', 'streamspot']
    datasets = ['cadets', 'trace', 'theia', 'optc_1', 'optc_2', 'optc_3', 'streamspot']
    # bss = [512,1024,2048]
    bss = [128, 256, 512, 1024]
    # svds = [0,1]
    svds = [0]
    bu = [0, 1]
    epochs = [50, 100, 150, 200]

    for gnn in ['gcn', 'gin', 'gnn', 'graphsage']:
        for dataset in ['cadets','streamspot','optc_1', 'optc_2', 'optc_3']:

            dataset_args = copy.deepcopy(args)
            dataset_args.dataset = dataset
            dataset_args.batch_size = 512
            dataset_args.bu = 1
            dataset_args.svd = 0
            dataset_args.max_epoch = 50
            dataset_args.gnn = gnn

            task_combinations = get_all_task_combinations()
            total_combinations = len(task_combinations)

            with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
                for i, tasks in enumerate(task_combinations, 1):
                    if len(tasks) < 6: continue

                    run_args = copy.deepcopy(dataset_args)
                    run_args.tasks = tasks

                    main(run_args)
    # ablation of the epochs
    for e in epochs:
        for dataset in datasets:

            dataset_args = copy.deepcopy(args)
            dataset_args.dataset = dataset
            dataset_args.batch_size = 512
            dataset_args.bu = 1
            dataset_args.svd = 0
            dataset_args.max_epoch = e

            task_combinations = get_all_task_combinations()
            total_combinations = len(task_combinations)

            with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
                for i, tasks in enumerate(task_combinations, 1):
                    if len(tasks) < 6: continue

                    run_args = copy.deepcopy(dataset_args)
                    run_args.tasks = tasks

                    main(run_args)
    # ablation of the batch size
    for bs in bss:
        for dataset in datasets:

            dataset_args = copy.deepcopy(args)
            dataset_args.dataset = dataset
            dataset_args.batch_size = bs
            dataset_args.bu = 1
            dataset_args.svd = 0
            dataset_args.max_epoch = 50

            task_combinations = get_all_task_combinations()
            total_combinations = len(task_combinations)

            with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
                for i, tasks in enumerate(task_combinations, 1):
                    if len(tasks) < 6: continue

                    run_args = copy.deepcopy(dataset_args)
                    run_args.tasks = tasks

                    main(run_args)

    # ablation of the pretext tasks
    for dataset in datasets:

        dataset_args = copy.deepcopy(args)
        dataset_args.dataset = dataset
        dataset_args.batch_size = 512
        dataset_args.bu = 1
        dataset_args.svd = 0
        dataset_args.max_epoch = 50

        task_combinations = get_all_task_combinations()
        total_combinations = len(task_combinations)

        with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
            for i, tasks in enumerate(task_combinations, 1):
                if len(tasks) < 5: continue

                run_args = copy.deepcopy(dataset_args)
                run_args.tasks = tasks

                main(run_args)
    # ablation of the bu
    for b in bu:

        dataset_args = copy.deepcopy(args)
        dataset_args.dataset = dataset
        dataset_args.batch_size = 512
        dataset_args.bu = b
        dataset_args.svd = 0
        dataset_args.max_epoch = 50

        task_combinations = get_all_task_combinations()
        total_combinations = len(task_combinations)

        with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
            for i, tasks in enumerate(task_combinations, 1):
                if len(tasks) < 6: continue

                run_args = copy.deepcopy(dataset_args)
                run_args.tasks = tasks

                main(run_args)

def tune_iter(args):
    # datasets = ['cadets', 'trace', 'theia','optc_1','optc_2','optc_3', 'wget', 'streamspot']
    # datasets = ['cadets', 'theia', 'optc_1', 'optc_2', 'optc_3', 'streamspot','trace']
    datasets = ['wget']
    # ds = [32, 64, 128, 256, 512]
    ds = [32, 64, 128, 256]
    ls = [1, 2, 3, 4, 5]
    for l in tqdm(ls):
        print("========================================= l: ", l)
        for dataset in datasets:

            args.d = 64
            args.l = l
            dataset_args = copy.deepcopy(args)
            dataset_args.dataset = dataset

            task_combinations = get_all_task_combinations()
            total_combinations = len(task_combinations)

            with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
                for i, tasks in enumerate(task_combinations, 1):
                    if len(tasks) < 6: continue

                    run_args = copy.deepcopy(dataset_args)
                    run_args.tasks = tasks

                    main(run_args)
    for d in tqdm(ds):
        print("========================================= d: ", d)
        for dataset in datasets:
            # 深拷贝参数以避免相互影响
            args.d = d
            args.l = 3
            dataset_args = copy.deepcopy(args)
            dataset_args.dataset = dataset

            task_combinations = get_all_task_combinations()
            total_combinations = len(task_combinations)

            with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
                for i, tasks in enumerate(task_combinations, 1):
                    if len(tasks) <6: continue

                    run_args = copy.deepcopy(dataset_args)
                    run_args.tasks = tasks

                    main(run_args)


def scaling_data_iter(args):

    for train_g in [1,2,3,4]:
        print("========================================= train_g number: ", train_g)
        for dataset in ['cadets','theia','trace']:
            args.d = 64
            args.l = 3
            dataset_args = copy.deepcopy(args)
            dataset_args.dataset = dataset
            dataset_args.train_g_number = train_g

            task_combinations = get_all_task_combinations()
            total_combinations = len(task_combinations)
            with tqdm(total=total_combinations, desc=f"Dataset: {dataset}") as pbar:
                for i, tasks in enumerate(task_combinations, 1):
                    if len(tasks) < 6: continue

                    run_args = copy.deepcopy(dataset_args)
                    run_args.tasks = tasks

                    main(run_args)

def iterate_main(args):
    if args.iterate_train == 1 and args.param_tune == 0 and args.iterate_scaling_data == 0:
        all_iter(args)
    if args.param_tune == 1:
        tune_iter(args)
    if args.iterate_scaling_data == 1:
        scaling_data_iter(args)

def sample_graphs(subgraph, new_seed_id, hop):

    neis = set([new_seed_id])
    for i in range(hop):
        if i == 0:
            tmp_neis = [new_seed_id]
        for nei in tmp_neis:

            cur_neis = set(subgraph.successors(nei).tolist() + subgraph.predecessors(nei).tolist())
            neis = neis.union(cur_neis)

            tmp_neis = cur_neis

    return torch.tensor(list(neis))

all_loss = []
def show_avg_loss():
    avg_loss = {}
    for l in all_loss:
        for k, v in l.items():
            if k not in avg_loss:
                avg_loss[k] = []
            avg_loss[k].append(v)
    output = ""
    for k, v in avg_loss.items():
        output += f"{k}: {sum(v) / len(v):.6f}, "
    print(output)


def add_loss(total_loss):
    # store the cpu value
    loss = {k: v.detach().cpu().item() for k, v in total_loss.items()}
    all_loss.append(loss)


def main(main_args):
    print("device: ", main_args.device)
    main_args.device = device = f'cuda:{main_args.device}' if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset
    if dataset_name == 'streamspot':
        main_args.max_epoch = 5
        main_args.num_hidden = 256
        main_args.num_layers = 4
    elif dataset_name == 'wget':
        main_args.max_epoch = 2
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        # main_args.max_epoch = 100
        # main_args.max_epoch = 50
        main_args.num_hidden = main_args.d
        main_args.num_layers = main_args.l

    set_random_seed(0)

    if main_args.param_tune == 0:
        if args.iterate_scaling_data == 1:
            ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_traing_number{main_args.train_g_number}.pt")
        elif args.gnn == 'gat':
            ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_{'_'.join(main_args.tasks)}"
                       f"_epoch{main_args.max_epoch}_bu{main_args.bu}_svd{main_args.svd}_bs{main_args.batch_size}.pt")
        else:
            ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_{'_'.join(main_args.tasks)}"
                       f"_epoch{main_args.max_epoch}_bu{main_args.bu}_svd{main_args.svd}_bs{main_args.batch_size}_{main_args.gnn}.pt")
    else:

        main_args.num_hidden = main_args.d
        main_args.num_layers = main_args.l
        ck_name = (f"./checkpoints/{dataset_name}/checkpoint-{dataset_name}_d{args.d}_l{args.l}.pt")

    if os.path.exists(ck_name):
        print("Trained model exists: ", ck_name)
        # return
    else:
        print("Begin to generate training ck:", ck_name)

        # with open(ck_name, 'w') as f:
        #     pass

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        if dataset_name == 'streamspot':
            batch_size = 12
        else:
            batch_size = 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model = model.to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        # model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
        #                           optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim)
        train_loader = extract_dataloaders(train_index, batch_size)
        epoch_iter = tqdm(range(main_args.max_epoch))
        for epoch in epoch_iter:
            model.train()
            loss_list = []
            for _, batch in enumerate(train_loader):
                batch_g = [transform_graph(graphs[idx][0], main_args.n_dim, main_args.e_dim).to(main_args.device) for idx in batch]
                batch_g = dgl.batch(batch_g)
                model.train()
                total_loss = model(batch_g)
                # add loss value to list
                add_loss(total_loss)
                # dynamic task priority
                losses = []
                for k, v in total_loss.items():
                    losses.append(getattr(main_args, k.replace('loss', 'cof')) * v)
                losses = torch.stack(losses)
                # losses = torch.stack(list(total_loss.values()))
                if main_args.bu == 1:
                    final_loss, _ = model.bayes_weighting(losses)
                else:
                    final_loss = losses.sum()
                loss_list.append(final_loss.item())

                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
                del batch_g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

        torch.save(model.state_dict(), ck_name)
        # torch.save(model.state_dict(), "./checkpoints/checkpoint-{}_{}_{}.pt".format(dataset_name,main_args.num_hidden,main_args.num_layers))
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        # print the parameters
        print(main_args)

        model = build_model(main_args)
        model = model.to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        epoch_iter = tqdm(range(main_args.max_epoch))
        n_train = metadata['n_train']

        start_time = time.time()
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                if main_args.iterate_scaling_data == 1 and i > main_args.train_g_number - 1:
                    continue
                model.train()
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                # record memory cost
                if epoch %10 == 0 and epoch > 0:  print_mem(epoch)
                  # =======================================================
                optimizer.zero_grad()
                total_loss = model(g)
                # add loss value to list
                add_loss(total_loss)
                # dynamic task priority
                # losses = torch.stack(list(total_loss.values()))
                losses = []
                for k, v in total_loss.items():
                    losses.append(getattr(main_args, k.replace('loss','cof')) * v)
                losses = torch.stack(losses)
                if main_args.bu == 1:
                    final_loss, _ = model.bayes_weighting(losses)
                    final_loss /= n_train
                else:
                    final_loss = losses.sum() / n_train

                epoch_loss += final_loss.item()
                final_loss.backward()
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")

            show_avg_loss()
        torch.save(model.state_dict(), ck_name)
        end_time = time.time()

        # output trianing time in seconds
        print(f"Training time: {end_time - start_time} seconds")
        # save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)
        # os.unlink(save_dict_path)
    return

if __name__ == '__main__':
    args = build_args()

    if not args.iterate_train:
        main(args)
    else:
        iterate_main(args)
