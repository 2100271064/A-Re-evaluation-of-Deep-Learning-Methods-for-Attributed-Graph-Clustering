import random

import numpy as np
import torch
import sklearn
# from sklearn.preprocessing import normalize
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from networkx.readwrite import json_graph
import json

# from torch_geometric.datasets import Planetoid


# def get_dataset(dataset):
#     datasets = Planetoid('./dataset', dataset)
#     return datasets
#
# def data_preprocessing(dataset):
#     dataset.adj = torch.sparse_coo_tensor(
#         dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
#     ).to_dense()
#     dataset.adj_label = dataset.adj
#
#     dataset.adj += torch.eye(dataset.x.shape[0])
#     dataset.adj = normalize(dataset.adj, norm="l1")
#     dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)
#
#     return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = sklearn.preprocessing.normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def cal_n_cluster(dataset):
    # return [cluster_num, outliers_num]
    if dataset == 'cora':
        return [18, 7]
    elif dataset == 'citeseer':
        return [12, 1]
    elif dataset == 'pubmed':
        return [26, 0]
    elif dataset == 'wiki':
        return [17, 6]
    elif dataset == 'acm':
        return [12, 0]
    elif dataset == 'dblp':
        return [16, 0]
    elif dataset == 'uk' or dataset == 'us':
        return [20, 0]

def load_data(datasets):

    root_path = r"/home/laixy/AHG/dataset/{0}".format(datasets)
    # 抽取需要的
    node = np.array(pd.read_table(root_path + r"/node.txt", header=None)[0])

    if datasets != 'uk' and datasets != 'us':
        # 全部顶点的特征矩阵
        features = np.array(pd.read_table(root_path + r"/feature.txt", sep=" ", header=None))
        features = features[node]

        # 全部顶点的邻接矩阵
        adj = np.array(pd.read_table(root_path + r"/adj.txt", sep=" ", header=None))

        adj = adj[node]
        adj = adj[:, node]
        adj_label = adj

        adj = adj + np.eye(features.shape[0])
        adj = normalize(adj, norm="l1")
    else:
        features = np.load(root_path + r"/feat192.npy")
        features = features[node]

        nx_graph = json_graph.node_link_graph(json.load(open(root_path + '/(renumberated)graph.json')))
        adj = nx.to_scipy_sparse_matrix(nx_graph)
        adj_label = adj

        adj = adj + sp.eye(adj.shape[0])
        adj = normalize(adj)

    labels = np.array(pd.read_table(root_path + r"/label.txt", header=None)[0])

    return features, labels, adj_label, adj

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