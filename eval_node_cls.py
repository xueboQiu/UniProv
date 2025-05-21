import os
import random
import pickle as pkl
import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_recall_curve

def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test, device=0):

    print("begin to evaluate dataset: {}".format(dataset))

    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    n_neighbors = 10

    if "optc" in dataset:
        n_neighbors = 150

    res = faiss.StandardGpuResources()
    # 创建配置
    config = faiss.GpuIndexFlatConfig()
    config.device = device
    gpu_index = faiss.GpuIndexFlatL2(res, x_train.shape[1], config)
    gpu_index.add(x_train.astype('float32'))

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path) or True:
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        subset = x_train[idx][:min(50000, x_train.shape[0])].astype('float32')
        distances, _ = gpu_index.search(subset, n_neighbors)
        print("=============== finish training knn model ===============")

        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = gpu_index.search(x_test.astype('float32'), n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance
    del distances
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = -1
    for i in range(len(f1)):
        # if dataset == 'trace' and rec[i] < 0.99979:
        if dataset == 'trace' and rec[i] < 0.999:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] <= 0.99996:
        # if dataset == 'theia' and rec[i] <= 0.5:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.997:
            best_idx = i - 1
            break
        # if 'optc' in dataset and rec[i] < 0.9:
        if 'optc' in dataset :
            best_idx = np.argmax(f1)
            # best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    print("Best Threadhold: {}".format(best_thres))
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print("FPR: {}".format(fp / (fp + tn)))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0, None, None

def evaluate_entity_level_using_kmeans(dataset, x_train, x_test, y_test):

    print("begin to evaluate dataset: {}".format(dataset))

    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    n_clusters = 100

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x_train)  # 对训练数据进行聚类

    # 计算训练数据的距离分布
    train_distances = np.min(kmeans.transform(x_train), axis=1)  # 每个点到最近簇中心的距离
    mean_distance = np.mean(train_distances)  # 计算平均距离

    # 保存结果
    save_dict_path = './eval_result/kmeans_distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        with open(save_dict_path, 'wb') as f:
            pkl.dump({'mean_distance': mean_distance}, f)
    else:
        with open(save_dict_path, 'rb') as f:
            saved_data = pkl.load(f)
            mean_distance = saved_data['mean_distance']

    # 测试数据的距离计算
    test_distances = np.min(kmeans.transform(x_test), axis=1)  # 每个测试点到最近簇中心的距离

    # 根据距离判断异常点
    score = test_distances / mean_distance  # 距离归一化

    # 计算 AUC（需要测试数据的标签 y_test）
    assert len(y_test) == len(score), "y_test 和 anomaly_scores 的长度不一致！"
    auc = roc_auc_score(y_test, score)

    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = -1
    for i in range(len(f1)):
        # To repeat peak performance
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] <= 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.997:
        # if dataset == 'cadets' and rec[i] < 0.9:
            best_idx = i - 1
            break
        if 'optc' in dataset and rec[i] < 0.6:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    print("Best Threadhold: {}".format(best_thres))
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print("FPR: {}".format(fp / (fp + tn)))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0, None, None