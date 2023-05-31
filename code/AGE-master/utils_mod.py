import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.preprocessing as preprocess
import pandas as pd
from networkx.readwrite import json_graph
import json

# def sample_mask(idx, l):
#     """Create mask."""
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask, dtype=np.bool)

def load_data(dataset):

    root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)

    # 抽取需要的
    node = np.array(pd.read_table(root_path + r"/node.txt", header=None)[0])

    if (dataset != "us" and dataset != "uk"):

        # 全部顶点的特征矩阵
        features = np.array(pd.read_table(root_path + r"/feature.txt", sep=" ", header=None))
        features = features[node]
        # 全部顶点的邻接矩阵
        adj = np.array(pd.read_table(root_path + r"/adj.txt", sep=" ", header=None))
        adj = adj[node]
        adj = adj[:, node]
        # adj.type = scipy.sparse.csr.csr_matrix
        adj = sp.csr_matrix(adj)
    else:
        features = np.load(root_path + r"/feat192.npy")
        features = features[node]
        nx_graph = json_graph.node_link_graph(json.load(open(root_path + r"/(renumberated)graph.json")))
        adj = nx.to_scipy_sparse_matrix(nx_graph, format='csr')

    features = torch.FloatTensor(features)
    labels = np.array(pd.read_table(root_path + r"/label.txt", header=None)[0])

    return adj, features, labels

# def load_wiki():
#     f = open('data/graph.txt','r')
#     adj, xind, yind = [], [], []
#     for line in f.readlines():
#         line = line.split()
        
#         xind.append(int(line[0]))
#         yind.append(int(line[1]))
#         adj.append([int(line[0]), int(line[1])])
#     f.close()
#     ##print(len(adj))

#     f = open('data/group.txt','r')
#     label = []
#     for line in f.readlines():
#         line = line.split()
#         label.append(int(line[1]))
#     f.close()

#     # node_id feature_id 边权重
#     f = open('data/tfidf.txt','r')
#     fea_idx = []
#     fea = []
#     adj = np.array(adj)
#     # 双向？
#     adj = np.vstack((adj, adj[:,[1,0]]))
#     # 去除重复，并按照从小到大排序
#     adj = np.unique(adj, axis=0)
    
#     labelset = np.unique(label)
#     print("label.unique", len(labelset))
#     # 给label重新编号
#     labeldict = dict(zip(labelset, range(len(labelset))))
#     label = np.array([labeldict[x] for x in label])
#     # 邻接矩阵稀疏化
#     adj = sp.csr_matrix((np.ones(len(adj)), (adj[:,0], adj[:,1])), shape=(len(label), len(label)))

#     for line in f.readlines():
#         line = line.split()
#         fea_idx.append([int(line[0]), int(line[1])])
#         fea.append(float(line[2]))
#     f.close()

#     fea_idx = np.array(fea_idx)
#     features = sp.csr_matrix((fea, (fea_idx[:,0], fea_idx[:,1])), shape=(len(label), 4973)).toarray()
#     # feature做归一化处理
#     scaler = preprocess.MinMaxScaler()
#     #features = preprocess.normalize(features, norm='l2')
#     features = scaler.fit_transform(features)
#     features = torch.FloatTensor(features)

#     return adj, features, label


# def parse_index_file(filename):
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index


# def sparse_to_tuple(sparse_mx):
#     if not sp.isspmatrix_coo(sparse_mx):
#         sparse_mx = sparse_mx.tocoo()
#     coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#     values = sparse_mx.data
#     shape = sparse_mx.shape
#     return coords, values, shape
#
#
# def mask_test_edges(adj):
#     # Function to build test set with 10% positive links
#     # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
#     # TODO: Clean up.
#
#     # Remove diagonal elements
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#     adj.eliminate_zeros()
#     # Check that diag is zero:
#     assert np.diag(adj.todense()).sum() == 0
#
#     adj_triu = sp.triu(adj)
#     adj_tuple = sparse_to_tuple(adj_triu)
#     edges = adj_tuple[0]
#     edges_all = sparse_to_tuple(adj)[0]
#     num_test = int(np.floor(edges.shape[0] / 10.))
#     num_val = int(np.floor(edges.shape[0] / 20.))
#
#     all_edge_idx = list(range(edges.shape[0]))
#     np.random.shuffle(all_edge_idx)
#     val_edge_idx = all_edge_idx[:num_val]
#     test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#     test_edges = edges[test_edge_idx]
#     val_edges = edges[val_edge_idx]
#     train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
#
#     def ismember(a, b, tol=5):
#         rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
#         return np.any(rows_close)
#
#     test_edges_false = []
#     while len(test_edges_false) < len(test_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if test_edges_false:
#             if ismember([idx_j, idx_i], np.array(test_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(test_edges_false)):
#                 continue
#         test_edges_false.append([idx_i, idx_j])
#
#     val_edges_false = []
#     while len(val_edges_false) < len(val_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], train_edges):
#             continue
#         if ismember([idx_j, idx_i], train_edges):
#             continue
#         if ismember([idx_i, idx_j], val_edges):
#             continue
#         if ismember([idx_j, idx_i], val_edges):
#             continue
#         if val_edges_false:
#             if ismember([idx_j, idx_i], np.array(val_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(val_edges_false)):
#                 continue
#         val_edges_false.append([idx_i, idx_j])
#
#     #assert ~ismember(test_edges_false, edges_all)
#     #assert ~ismember(val_edges_false, edges_all)
#     #assert ~ismember(val_edges, train_edges)
#     #assert ~ismember(test_edges, train_edges)
#     #assert ~ismember(val_edges, test_edges)
#
#     data = np.ones(train_edges.shape[0])
#
#     # Re-build adj matrix
#     adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T
#
#     # NOTE: these edge lists only contain single direction of edge!
#     return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# def decompose(adj, dataset, norm='sym', renorm=True):
#     adj = sp.coo_matrix(adj)
#     ident = sp.eye(adj.shape[0])
#     if renorm:
#         adj_ = adj + ident
#     else:
#         adj_ = adj
#
#     rowsum = np.array(adj_.sum(1))
#
#     if norm == 'sym':
#         degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#         adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#         laplacian = ident - adj_normalized
#     evalue, evector = np.linalg.eig(laplacian.toarray())
#     np.save(dataset + ".npy", evalue)
#     print(max(evalue))
#     exit(1)
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     n, bins, patches = ax.hist(evalue, 50, facecolor='g')
#     plt.xlabel('Eigenvalues')
#     plt.ylabel('Frequncy')
#     fig.savefig("eig_renorm_" + dataset + ".png")
#

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

# def laplacian(adj):
#     rowsum = np.array(adj.sum(1))
#     degree_mat = sp.diags(rowsum.flatten())
#     lap = degree_mat - adj
#     return torch.FloatTensor(lap.toarray())
#
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


# def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     # Predict on test set of edges
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])
#
#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])
#
#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#
#     return roc_score, ap_score

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

# if __name__ == '__main__':
    # names = ["cora", "citeseer", "pubmed"]
    # for i in range(len(names)):
    #     print("dataset:", names[i])
    #     adj, features, labels, idx_train, idx_val, idx_test = load_data(names[i])
    #     print("adj:", adj.shape)
    #     adj_arr = adj.toarray()
    #     print("edge:", adj_arr.sum(0).sum())
    #     print("对称") if (adj_arr == adj_arr.T).all() else print("不对称")
    #     print("features:", features.shape)
    #     label_unique = np.unique(labels)
    #     print("labels:", label_unique)

    # adj, features, labels = load_wiki()
    # print("adj:", adj.shape)
    # adj_arr = adj.toarray()
    # print("edge:", adj_arr.sum(0).sum())
    # print("对称") if (adj_arr == adj_arr.T).all() else print("不对称")
    # print("features:", features.shape)
    # label_unique = np.unique(labels)
    # print("labels:", label_unique)
