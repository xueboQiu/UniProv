import argparse


def build_task_list(args):
    task_mapping = {
        'p_link': args.p_link,
        'p_recon': args.p_recon,
        'p_ming': args.p_ming,
        'p_minn': args.p_minn,
        'p_misgsg': args.p_misgsg,
        'p_decor': args.p_decor,
    }
    return [task for task, enabled in task_mapping.items() if enabled]
def build_args():
    parser = argparse.ArgumentParser(description="MAGIC")
    parser.add_argument("--dataset", type=str, default="cadets")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gnn", type=str, default="gat")
    # parser.add_argument("--gnn", type=str, default="gcn")
    # parser.add_argument("--gnn", type=str, default="gin")
    # parser.add_argument("--gnn", type=str, default="gnn")
    # parser.add_argument("--gnn", type=str, default="graphsage")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--alpha_l", type=float, default=3, help="`pow`index for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss_fn", type=str, default='sce')
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument('--snapshots', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--temp", type=float, required=False, default=0.07)
    parser.add_argument("--alpha", type=float, required=False, default=0.3)
    # parser.add_argument("--beta", type=float, required=False, default=0.7)
    parser.add_argument("--d", type=int, required=False, default=64)
    parser.add_argument("--l", type=int, required=False, default=3)
    # foundation param
    parser.add_argument("--fanout", type=str, help="fanout numbers", default='5,5')
    parser.add_argument('--use_saint', action='store_true')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--decor_der', type=float, default=0.1, help='der for decor')
    parser.add_argument('--decor_dfr', type=float, default=0.1, help='dfr for decor')
    parser.add_argument('--decor_size', type=int, default=3000, help='size for subgraphs used in decor')
    parser.add_argument('--minsg_der', type=float, default=0.3, help='drop edge ratio for minsg')
    parser.add_argument('--minsg_dfr', type=float, default=0.3, help='drop feature ratio for minsg')
    parser.add_argument('--gm_edge_drop_ratio', type=float, default=0.2, help='edge perturbation ratio')
    parser.add_argument('--gm_node_drop_ratio', type=float, default=0.2, help='node perturbation ratio')
    parser.add_argument('--gm_feat_drop_ratio', type=float, default=0.2, help='feature perturbation ratio')
    parser.add_argument('--khop_ming', type=int, default=3, help='order for ming sampling')
    parser.add_argument('--khop_minsg', type=int, default=2, help='order for minsg sampling')
    parser.add_argument("--batch_size_sampling", default=5, type=int, help="Batch size for training graph samping.")
    parser.add_argument("--batch_size_multiplier_minsg", default=10, type=int, help="Batch size multiplier for minsg.")

    parser.add_argument("--batch_size_multiplier_ming", default=10, type=int, help="Batch size multiplier for ming.")
    parser.add_argument("--per_gpu_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument('--sub_size', type=int, default=30, help='size for subgraphs used in gm')
    parser.add_argument('--lp_neg_ratio', type=int, default=1, help='negative ratio for link prediction pretrain')
    parser.add_argument("--worker", type=int, default=3, help="number of workers for dataloader")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")
    parser.add_argument('--predictor_dim', type=int, default=64)
    parser.add_argument('--temperature_minsg', type=float, default=0.1, help='minsg: tau for infoNCE')
    parser.add_argument('--threshold', type=float, default=0.9, help='threshold used in graph matching task')
    # iterate all the tasks
    parser.add_argument('--bu', type=int, default=1, help='using adaptive loss weight adjustments.')
    parser.add_argument('--svd', type=int, default=0, help='augmentation excluding articulation nodes and bridge edges.')
    parser.add_argument("--iterate_train", default=0, type=int)
    parser.add_argument("--param_tune", default=0, type=int, help='iterate parameter tuning.')
    parser.add_argument("--iterate_eval", default=0, type=int)
    parser.add_argument("--iterate_eval_params", default=0, type=int)
    parser.add_argument("--iterate_eval_scaling", default=0, type=int)
    parser.add_argument("--iterate_abl", default=0, type=int)
    parser.add_argument("--iterate_scaling_data", default=0, type=int)
    parser.add_argument("--train_g_number", default=0, type=int)
    # different tasks tags
    parser.add_argument("--p_link", default=1, type=int)
    parser.add_argument("--p_recon", default=1, type=int)
    parser.add_argument("--p_ming", default=1, type=int)
    parser.add_argument("--p_minn", default=1, type=int)
    parser.add_argument("--p_misgsg", default=1, type=int)
    parser.add_argument("--p_decor", default=1, type=int)
    # coefficient for different tasks
    parser.add_argument('--misgsg_cof', type=float, default=10)
    parser.add_argument('--ming_cof', type=float, default=10)
    # parser.add_argument('--minn_cof', type=float, default=1)
    # parser.add_argument('--misgsg_cof', type=float, default=1)
    # parser.add_argument('--ming_cof', type=float, default=1)
    parser.add_argument('--minn_cof', type=float, default=10)
    parser.add_argument('--decor_cof', type=float, default=1)
    parser.add_argument('--link_cof', type=float, default=1)
    parser.add_argument('--recon_cof', type=float, default=1)

    parser.add_argument('--dt_node_cls', type=int, default=0, help='downstream task of node classification')
    parser.add_argument('--dt_graph_match', type=int, default=0, help='downstream task of graph match')
    parser.add_argument('--dt_graph_cls', type=int, default=0, help='downstream task of graph classification')
    parser.add_argument('--dt_node_clt', type=int, default=0, help='downstream task of node cluster')

    args = parser.parse_args()
    args.tasks = build_task_list(args)
    args.downstream_tasks = [task for task in ['dt_node_cls', 'dt_graph_match', 'dt_graph_cls', 'dt_node_clt'] if getattr(args, task)]
    return args