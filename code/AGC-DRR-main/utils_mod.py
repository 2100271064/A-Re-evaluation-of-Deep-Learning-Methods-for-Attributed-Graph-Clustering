import torch
import random
import numpy as np
import scipy.sparse as sp
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


# def adjust_learning_rate(optimizer, epoch):
#     lr = 0.001 * (0.1 ** (epoch // 50))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        print("真实类：{}个,预测类仅有:{}个！".format(numclass1, numclass2))
        return -1, -1

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, n_outliers, y_pred, epoch=0):

    # 剥离outliers
    if n_outliers > 0:
        y_true = y_true[:-n_outliers]
        y_pred = y_pred[:-n_outliers]

    acc, f1 = cluster_acc(y_true, y_pred)
    if acc == -1:
        return -1, -1, -1, -1
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


# def parameter(model):
#     params = list(model.parameters())
#     k = 0
#     for i in params:
#         l = 1
#         for j in i.size():
#             l *= j
#         k = k + l
#     print("sum:" + str(k))
#     return str(k)

# def target_distribution(q):
#     weight = q ** 2 / q.sum(0)
#     return (weight.t() / weight.sum(1)).t()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 归一化邻接矩阵
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def cal_n_cluster(dataset):
    # return [cluster_num, outliers_num]
    if dataset == 'cora':
        return [18, 7]
    elif dataset == 'citeseer':
        return [12, 1]
    elif dataset == 'pubmed':
        return [26, 0]
    elif dataset == 'wiki':
        return [17, 5]
    elif dataset == 'acm':
        return [12, 0]
    elif dataset == 'dblp':
        return [17, 0]
    elif dataset == 'uk' or dataset == 'us':
        return [20, 0]
