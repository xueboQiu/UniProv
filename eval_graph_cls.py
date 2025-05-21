import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors

from utils.loaddata import load_batch_level_dataset, transform_graph
from utils.utils import set_random_seed


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    n_neighbors = min(int(train_count * 0.02), 10)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / x_train_std
            x_test = (x_test - x_train_mean) / x_train_std

            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            f1 = 2 * prec * rec / (rec + prec + 1e-9)
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

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
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))
        return np.mean(auc_list), np.std(auc_list)
    else:
        set_random_seed(2022)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
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
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1[best_idx]))
        print('PRECISION: {}'.format(prec[best_idx]))
        print('RECALL: {}'.format(rec[best_idx]))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        return auc, 0.0