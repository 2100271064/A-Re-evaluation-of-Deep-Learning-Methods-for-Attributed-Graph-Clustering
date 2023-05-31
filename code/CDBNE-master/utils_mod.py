import numpy as np
import sklearn
import torch
import random
# from sklearn.preprocessing import normalize
# from torch_geometric.datasets import Planetoid

import scipy.sparse as sp
import pandas as pd

# def get_dataset(dataset):
#     datasets = Planetoid(root='./dataset', name=dataset)
#     # print("训练集节点数量",sum(datasets.data.train_mask))
#     return datasets
#
# def data_preprocessing(dataset):
#     # 其实就是用边构建邻接矩阵，参考 https://pytorch.apachecn.org/docs/1.0/torch_tensors.html
#     dataset.adj = torch.sparse_coo_tensor(
#         dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
#     ).to_dense()
#     dataset.adj_label = dataset.adj
#
#     # torch.eye: 返回二维张量，对角线上是1，其它地方是0.
#     # 给邻接矩阵加上节点到自己的边
#     dataset.adj += torch.eye(dataset.x.shape[0])
#     # 每个元素除以每行的l1范数，即每行元素和，如果是l2就是除以每行样本的l2范数
#     # 这里的adj就是论文中的 transition matrix B_{ij}=1/d_i if e_{ij} \in E
#     dataset.adj = normalize(dataset.adj, norm="l1")
#     dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)
#
#     return dataset

def setup(args):
    """
    setup
    Return: None

    """
    args.lr = 0.0001
    [args.n_clusters, args.n_outliers] = cal_n_cluster(args.name)

    # GPU是否开启
    if torch.cuda.is_available() and args.cuda:
        print("Available GPU")
        args.device = torch.device("cuda")
    else:
        print("Using CPU")
        args.device = torch.device("cpu")

    print("------------------------------")
    print("dataset       : {}".format(args.name))
    print("device        : {}".format(args.device))
    print("clusters      : {}".format(args.n_clusters))
    print("learning rate : {:.0e}".format(args.lr))
    print("upd           : {}".format(args.update_interval))
    print("epoch        : {}".format(args.max_epoch))
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
    # adj = sp.csr_matrix(adj)


    adj_normalized = sp.coo_matrix(adj)
    adj_normalized = adj_normalized + sp.eye(adj_normalized.shape[0])
    adj_normalized = normalize(adj_normalized)
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    features = torch.FloatTensor(features)
    # adj.type = scipy.sparse.csr.csr_matrix

    return adj, adj_normalized, features, labels

def normalize(mx):
    rowsum = np.array(mx.sum(1))
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

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = sklearn.preprocessing.normalize(adj_numpy, norm="l1", axis=0)
    # M就是论文中的proximity matrix M
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

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


