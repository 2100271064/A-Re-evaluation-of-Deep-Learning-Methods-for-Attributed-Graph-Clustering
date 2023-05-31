#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License
import argparse
parser = argparse.ArgumentParser()
# acm、citeseer、cora、dblp、wiki、pubmed
parser.add_argument('--dataset', type=str, default='wiki', help='Dataset to use')
# parser.add_argument('--t1', type=int, default=1)
# parser.add_argument('--t2', type=int, default=50)
# parser.add_argument('--min_epoch_t', type=int, default=20)
args = parser.parse_args()

import numpy as np
import torch
import scipy.sparse as sp
from model_mod import ReGMM_VGAE
# from datasets import format_data
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
from sklearn.decomposition import PCA

# Dataset Name
dataset = args.dataset
print(dataset)
# 原来是 feas = format_data('citeseer', './data/Citeseer')
# feas = format_data('citeseer', '../data/Citeseer')
# num_nodes = feas['features'].size(0)
# num_features = feas['features'].size(1)
# nClusters = 6

# tpye(adj)：csr_matrix（只有0、1）
# tpye(features)：lil_matrix -> coo_matrix
# type(labels)：ndarray
# 原来是 adj, features, labels = load_data('citeseer', './data/Citeseer')
adj, features, labels, num_nodes, nClusters, n_outliers = load_data(dataset)
print("adj", adj.shape)
print("features", features.shape)
print("labels", labels.shape)
print("num_nodes", num_nodes)
print("nClusters", nClusters)
print("n_outliers", n_outliers)


# TODO 特征工程
if dataset != "pubmed":
    n_components = 500
    if (dataset == "dblp"):
        n_components = 256
    pca = PCA(n_components=n_components, svd_solver='full')
    features = pca.fit_transform(features)

# wiki数据集的features 数值太大，容易溢出
if dataset == "wiki":
    features = (features - np.min(features)) / (np.max(features) - np.min(features))

features = sp.coo_matrix(features)

num_features = features.shape[1]

# Network parameters
num_neurons = 32
embedding_size = 16
# 原来是 save_path = "./results/"
save_path = "/home/laixy/AHG/data/R-GAE-master/R-GMM-VGAE"

# Pretraining parameters
epochs_pretrain = 200
lr_pretrain = 0.01
# Clustering parameters
# 原来是200
epochs_cluster = 400

# lr_cluster 原来是注释部分
# lr_cluster = 0.001
# if dataset == 'cora':
#     lr_cluster = 0.01
lr_cluster = 0.0001

if dataset == 'acm':
    beta1 = 0.3
    beta2 = 0.15
elif dataset == 'citeseer':
    beta1 = 0.2
    beta2 = 0.1
elif dataset == 'cora':
    beta1 = 0.17 # TODO 原来是0.3
    beta2 = 0.08 # TODO 原来是0.15
elif dataset == 'dblp':
    beta1 = 0.14
    beta2 = 0.07
elif  dataset == 'wiki':
    beta1 = 0.08
    beta2 = 0.01
elif dataset == 'pubmed':
    beta1 = 0.1 # TODO 原来是0.4
    beta2 = 0.05 # TODO 原来是0.2

if dataset == 'citeseer':
    weight_decay = 0.089 # TODO 原来是0.089
    t1 = 1
    stable = 15 # beta1、bata2自适应调整
    b1 = 0.96 # beta1修改权重 # TODO 原来是0.96
    b2 = 0.98 # beta2修改权重 # TODO 原来是0.98
    t2 = 50
    min_epoch_t = 200
elif dataset == 'cora' or dataset == 'acm' or dataset == 'dblp' or dataset == 'wiki':
    weight_decay = 0.0001
    t1 = 10
    stable = 15
    b1 = 0.95
    b2 = 0.85
    t2 = 20
    min_epoch_t = 120
elif dataset == 'pubmed':
    weight_decay = 0.001
    t1 = 5
    stable = 20
    b1 = 0.95
    b2 = 0.85
    t2 = 50
    min_epoch_t = 100


# Data processing 
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features)
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2]))
weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig


# Training

run_round = 10
final_acc = []
final_nmi = []
final_ari = []
final_f1 = []
valid_epoch_num_list = []
for i in range(run_round):
    print('----------------------round_{0}-----------------------------'.format(i))

    network = ReGMM_VGAE(adj = adj_norm , num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="ReLU",
                         dataset=dataset)
    # -------------------------------------------------------
    # dense_tensor = features.to_dense()  # 转为密集矩阵
    # dense_tensor = dense_tensor / (dense_tensor.max() - dense_tensor.min()) * 10  # 将大于 3.0 的元素设为 10.0
    # values = dense_tensor[dense_tensor >= 0]  # 获取新的 values
    # indices = torch.nonzero(dense_tensor >= 0).t()  # 获取新的 indices
    # features = torch.sparse_coo_tensor(indices, values, dense_tensor.size())
    #
    # dense_tensor1 = adj.to_dense()  # 转为密集矩阵
    # dense_tensor1 = dense_tensor1 / (dense_tensor1.max() - dense_tensor1.min()) * 10  # 将大于 3.0 的元素设为 10.0
    # values = dense_tensor1[dense_tensor1 >= 0]  # 获取新的 values
    # indices = torch.nonzero(dense_tensor1 >= 0).t()  # 获取新的 indices
    # adj = torch.sparse_coo_tensor(indices, values, dense_tensor1.size())
    # --------------------------------------------------------
    if i == 0:
        network.pretrain(adj_norm, features, adj_label, labels, weight_tensor_orig, norm , epochs=epochs_pretrain, lr=lr_pretrain, save_path=save_path, dataset=dataset)

    mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = network.train(adj_norm, adj,  features, labels, n_outliers, norm, epochs=epochs_cluster, lr=lr_cluster, beta1=beta1, beta2=beta2, save_path=save_path, dataset=dataset,
                                                                           weight_decay=weight_decay, t1=t1, stable=stable, b1=b1, b2=b2, t2=t2, min_epoch_t=min_epoch_t)

    final_acc.append(mean_acc)
    final_nmi.append(mean_nmi)
    final_ari.append(mean_ari)
    final_f1.append(mean_f1)
    valid_epoch_num_list.append(valid_epoch_num)

acc_arr = np.array(final_acc)
nmi_arr = np.array(final_nmi)
ari_arr = np.array(final_ari)
f1_arr = np.array(final_f1)
print("{} epoch × 10, 有效的epoch数：".format(epochs_cluster), valid_epoch_num_list)

value = np.mean(acc_arr)
var = np.var(acc_arr)
std = np.std(acc_arr)
print('final_acc: {}, fianl_var_acc: {}, final_std_acc:{}'.format(value, var, std))
print('final_acc: {:.4f}, fianl_var_acc: {:.2f}, final_std_acc:{:.2f}%'.format(value, var, std * 100))

value = np.mean(nmi_arr)
var = np.var(nmi_arr)
std = np.std(nmi_arr)
print('final_nmi: {}, final_var_nmi: {}, final_std_nmi:{}'.format(value, var, std))
print('final_nmi: {:.4f}, final_var_nmi: {:.2f}, final_std_nmi:{:.2f}%'.format(value, var, std * 100))

value = np.mean(ari_arr)
var = np.var(ari_arr)
std = np.std(ari_arr)
print('final_ari: {}, final_var_ari: {}, final_std_ari:{}'.format(value, var, std))
print('final_ari: {:.4f}, final_var_ari: {:.2f}, final_std_ari:{:.2f}%'.format(value, var, std * 100))

value = np.mean(f1_arr)
var = np.var(f1_arr)
std = np.std(f1_arr)
print('final_f1: {}, final_var_f1: {}, final_std_f1:{}'.format(value, var, std))
print('final_f1: {:.4f}, final_var_f1: {:.2f}, final_std_f1:{:.2f}%'.format(value, var, std * 100))
