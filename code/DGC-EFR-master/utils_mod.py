import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import pandas as pd
import random

def load_graph(dataset):

    root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)
    adj = np.array(pd.read_table(root_path + r"/adj.txt", sep=" ", header=None))

    # 抽取需要的
    node = np.array(pd.read_table(root_path + r"/node.txt", header=None)[0])
    adj = adj[node]
    adj = adj[:, node]

    adj_normalized = sp.coo_matrix(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    adj_normalized = adj_normalized + sp.eye(adj_normalized.shape[0])
    adj_normalized = normalize(adj_normalized)
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj_normalized, adj_label


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
        root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)
        # 抽取需要的
        node = np.array(pd.read_table(root_path + r"/node.txt", header=None)[0])

        self.x = np.loadtxt(root_path + r'/feature.txt', dtype=float)
        self.x = self.x[node]
        self.y = np.loadtxt(root_path + r'/label.txt', dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))

# def create_exp_dir(path, scripts_to_save=None):
#   if not os.path.exists(path):
#     os.mkdir(path)
#   print('Experiment dir : {}'.format(path))
#
#   if scripts_to_save is not None:
#     os.mkdir(os.path.join(path, 'scripts'))
#     for script in scripts_to_save:
#       dst_file = os.path.join(path, 'scripts', os.path.basename(script))
#       shutil.copyfile(script, dst_file)

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