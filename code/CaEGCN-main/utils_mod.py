'''
Author: xinying Lai
Date: 2022-09-11 11:46:25
LastEditTime: 2022-09-11 18:27:27
Description: Do not edit
'''
import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import random
import networkx as nx
from networkx.readwrite import json_graph
import json


def load_graph(dataset):

    if dataset != 'uk' and dataset != 'us':
        adj = np.loadtxt('/home/laixy/AHG/dataset/{}/adj.txt'.format(dataset), dtype=np.float32)

        vertex_arr = np.loadtxt('/home/laixy/AHG/dataset/{}/node.txt'.format(dataset), dtype=int)
        adj = adj[vertex_arr]
        adj = adj[:, vertex_arr]

        adj = sp.coo_matrix(adj)
    else:
        nx_graph = json_graph.node_link_graph(json.load(open('/home/laixy/AHG/dataset/{}/(renumberated)graph.json'.format(dataset))))
        adj = nx.to_scipy_sparse_matrix(nx_graph, format='coo')

    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
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


class load_data(Dataset):
    def __init__(self, dataset):
        # 特征矩阵 shape = (n, dim)
        if dataset != 'uk' and dataset != 'us':
            self.x = np.loadtxt('/home/laixy/AHG/dataset/{}/feature.txt'.format(dataset), dtype=float)
        else:
            self.x = np.load('/home/laixy/AHG/dataset/{}/feat192.npy'.format(dataset), dtype=float)

        vertex_arr = np.loadtxt('/home/laixy/AHG/dataset/{}/node.txt'.format(dataset), dtype=int)
        self.x = self.x[vertex_arr]

        # 标签 shape = (n, 1)
        self.y = np.loadtxt('/home/laixy/AHG/dataset/{}/label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


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