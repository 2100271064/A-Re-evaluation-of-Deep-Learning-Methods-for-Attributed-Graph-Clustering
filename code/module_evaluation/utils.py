import opt
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup():
    """
    setup
    Return: None

    """
    print("setting:")
    setup_seed(opt.args.seed)

    # from DAEGC
    opt.args.lr = 0.0001
    [opt.args.n_clusters, opt.args.n_outliers] = cal_n_cluster(opt.args.name)

    opt.args.upth_ed = 0.001
    opt.args.lowth_st = 0.1
    opt.args.lowth_ed = 0.5
    if opt.args.name == 'acm' or opt.args.name == 'wiki':
        opt.args.gnnlayers = 1
        opt.args.upth_st = 0.0011
    elif opt.args.name == 'citeseer' or opt.args.name == 'dblp':
        opt.args.gnnlayers = 3
        opt.args.upth_st = 0.0015
    elif opt.args.name == 'cora':
        opt.args.gnnlayers = 8
        opt.args.upth_st = 0.0110
    # pubmed
    elif opt.args.name == 'pubmed':
        opt.args.gnnlayers = 35
        opt.args.upth_st = 0.0013
        opt.args.lowth_st = 0.7
        opt.args.lowth_ed = 0.8

    # GPU是否开启
    if torch.cuda.is_available() and opt.args.cuda:
        print("Available GPU")
        opt.args.device = torch.device("cuda")
    else:
        print("Using CPU")
        opt.args.device = torch.device("cpu")

    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("clusters      : {}".format(opt.args.n_clusters))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("gnnlayers       : {}".format(opt.args.gnnlayers))
    print("upth_st       : {}".format(opt.args.upth_st))
    print("upth_ed       : {}".format(opt.args.upth_ed))
    print("lowth_st       : {}".format(opt.args.lowth_st))
    print("lowth_ed       : {}".format(opt.args.lowth_ed))
    print("upd           : {}".format(opt.args.upd))
    print("epoch        : {}".format(opt.args.epoch))
    print("------------------------------")

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

def load_data(dataset):
    root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)
    # 全部顶点的特征矩阵
    features = np.array(pd.read_table(root_path + r"/feature.txt", sep=" ", header=None))
    # 全部顶点的邻接矩阵
    adj = np.array(pd.read_table(root_path + r"/adj.txt", sep=" ", header=None))
    labels = np.array(pd.read_table(root_path + r"/label.txt", header=None)[0])

    # 抽取需要的
    node = np.array(pd.read_table(root_path + r"/node.txt", header=None)[0])
    features = features[node]
    adj = adj[node]
    adj = adj[:, node]
    adj_label = adj

    # TODO from DFCN
    adj_ = sp.coo_matrix(adj)
    adj_ = adj_ + sp.eye(adj_.shape[0])
    adj_ = normalize(adj_)
    adj_ = sparse_mx_to_torch_sparse_tensor(adj_)

    features = torch.FloatTensor(features)
    # adj.type = scipy.sparse.csr.csr_matrix
    adj = sp.csr_matrix(adj)

    # return adj, features, labels
    return adj, adj_, adj_label, features, labels

def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# def update_similarity(z, upper_threshold, pos_num, A=None):
#     f_adj = np.matmul(z, np.transpose(z))
#     if A is not None:
#         print("A→S")
#         f_adj = f_adj * A
#     cosine = f_adj
#     cosine = cosine.reshape([-1, ])
#     pos_num = round(upper_threshold * len(cosine))
#     # neg_num = round((1 - lower_treshold) * len(cosine))
#
#     pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
#     # neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
#
#     return np.array(pos_inds)
#
# def update_threshold(upper_threshold, up_eta):
#     upth = upper_threshold + up_eta
#     # lowth = lower_treshold + low_eta
#     return upth
#
# # 负样本是除去正样本的那些（共有n(n+1)/2-len(pos)个）
# def cal_neg_inds(pos_inds, n):
#     pos_dict = dict(zip(pos_inds, np.ones(len(pos_inds))))
#     neg_inds = []
#     inds_arr = np.array(list(range(n*n))).reshape([n,n])
#     for i in range(n):
#         for j in range(i+1):
#             if pos_dict.get(inds_arr[i, j]) == None and pos_dict.get(inds_arr[j, i]) == None:
#                 neg_inds.append(inds_arr[i, j])
#             # else:
#             #     if pos_dict.get(inds_arr[i, j]) != None and pos_dict.get(inds_arr[j, i]) != None and i != j:
#     return np.array(neg_inds)

def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num, A=None):
    f_adj = np.matmul(z, np.transpose(z))
    if A is not None:
        print("A→S")
        f_adj = f_adj * A
    cosine = f_adj
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

def loss_function(preds, labels):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(preds, labels)
    return cost

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def target_distribution_from_emb(z):
    return F.softmax(z, dim=-1)

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

# 保存pos_inds.txt
# 格式：x y
def save_pos_inds(w_root_path, filename, pos_inds, n):
    path = w_root_path + r"save_pos_inds/" + filename + ".txt"
    x = pos_inds // n
    y = pos_inds % n
    data = np.hstack((x.reshape([-1, 1]), y.reshape([-1, 1])))
    np.savetxt(path, data, fmt="%d")

# 保存αij(ATT).txt
# N * N
def save_attention_alpha(w_root_path, filename, attention_alpha):
    path = w_root_path + r"save_attention_alpha/" + filename + ".txt"
    # attention_alpha.data.cpu().numpy()
    np.savetxt(path, attention_alpha.data.cpu().numpy(), fmt="%f")

# 构造拉普拉斯平滑器(多层)
def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    # 长度为n*n对角矩阵，对角值为1
    ident = sp.eye(adj.shape[0])
    if renorm:
        # A' = A + I
        adj_ = adj + ident
    else:
        adj_ = adj

    # D'
    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        # D'^(1/2)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        # D'^(1/2)A'D'^(1/2)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        # L' = I - D^(1/2)A'D^(1/2)
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    # 将原列表扩展成为layer个当前元素
    reg = [2/3] * (layer)

    adjs = []
    # layer层
    for i in range(len(reg)):
        # I - kL'
        adjs.append(ident-(reg[i] * laplacian))
    return adjs

# laplacian_smoothing
def laplacian_smoothing(adj, features):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # 构造拉普拉斯平滑器(多层)
    adj_norm_s = preprocess_graph(adj, opt.args.gnnlayers, norm='sym', renorm=True)
    # 存放平滑后的特征X'
    sm_fea_s = sp.csr_matrix(features).toarray()
    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    return adj, sm_fea_s

# 批处理函数!!!!!!!!!!!!!!!!!
def data_split(full_list, n_sample):
    offset = n_sample
    random.shuffle(full_list)
    len_all = len(full_list)
    index_now = 0
    split_list = []
    while index_now < len_all:
        # 0-2000
        if index_now+offset > len_all:
            split_list.append(full_list[index_now:len_all])
        else:
            split_list.append(full_list[index_now:index_now+offset])
        index_now += offset
    return split_list