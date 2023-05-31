#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License

import torch
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sklearn.preprocessing as preprocess
from networkx.readwrite import json_graph
import json

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_networks(dataset_str, data_path):
    """Read the data and preprocess the task information."""
    dataset_G = data_path+"{}-airports.edgelist".format(dataset_str)
    dataset_L = data_path+"labels-{}-airports.txt".format(dataset_str)
    label_raw, nodes = [], []
    with open(dataset_L, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            node, label = lines.split()
            if label == 'label': continue
            label_raw.append(int(label))
            nodes.append(int(node))
    label_raw = np.array(label_raw)
    print(label_raw)
    G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    
    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    features = np.zeros((degreeNode.size, degreeNode.max()+1))
    features[np.arange(degreeNode.size),degreeNode] = 1
    features = sp.csr_matrix(features)

    return adj, features, label_raw

# 修改了这里
def load_data(dataset):
    root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)

    # 抽取需要的
    node = np.loadtxt(root_path + r"/node.txt", dtype=int)

    if (dataset != "us" and dataset != "uk"):
        # 全部顶点的特征矩阵
        feature = np.loadtxt(root_path + r"/feature.txt", dtype=float)
        feature = feature[node]
        # 全部顶点的邻接矩阵
        adj = np.loadtxt(root_path + r"/adj.txt", dtype=int)
        adj = adj[node]
        adj = adj[:, node]
        adj = sp.csr_matrix(adj)
    else:
        feature = np.load(root_path + r"/feat192.npy")
        feature = feature[node]
        nx_graph = json_graph.node_link_graph(json.load(open(root_path + r"/(renumberated)graph.json")))
        adj = nx.to_scipy_sparse_matrix(nx_graph, format='csr')

    labels = np.loadtxt(root_path + r"/label.txt", dtype=int)

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

    [n_clusters, n_outliers] = cal_n_cluster(dataset)

    return adj, feature, labels, feature.shape[0], n_clusters, n_outliers


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_graph1(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false