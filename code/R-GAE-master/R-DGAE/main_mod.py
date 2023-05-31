#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed', help='Dataset to use')
# parser.add_argument('--t1', type=int, default=1)
# parser.add_argument('--t2', type=int, default=50)
# parser.add_argument('--min_epoch_t', type=int, default=20)
args = parser.parse_args()

import numpy as np
import torch
import scipy.sparse as sp
from model_mod import ReDGAE
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

features = sp.coo_matrix(features)

num_features = features.shape[1]

# Network parameters
alpha = 1.
gamma = 0.001
num_neurons = 32
embedding_size = 16
# 原来是 save_path = "./results/"
save_path = "/home/laixy/AHG/data/R-GAE-master/R-DGAE"

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

# 采样算子的第一置信度阈值，第一和第二置信度差值阈值
# beat1 取值选择 {0.1，0.2，0.3，0.4}
if dataset == 'acm':
    beta1 = 0.33
    beta2 = 0.18
elif dataset == 'citeseer':
    beta1 = 0.2
    beta2 = 0.1
elif dataset == 'cora':
    beta1 = 0.25 # TODO 原来是0.33
    beta2 = 0.125 # TODO 原来是0.18
elif dataset == 'dblp':
    beta1 = 0.20
    beta2 = 0.10
elif dataset == 'wiki':
    beta1 = 0.8
    beta2 = 0.4
elif dataset == 'pubmed':
    beta1 = 0.1 # TODO 原来是0.2
    beta2 = 0.05 # TODO 原来是0.1

if dataset == 'citeseer':
    t1 = 1 # 更新Ω的epoch周期
    t2 = 50 # 更新转化后的聚类子图的epoch周期
    min_epoch_t = 200 # 收敛标准|Ω| ≥ 0.9*|V|
elif dataset == 'cora' or dataset == 'acm' or dataset == 'dblp' or dataset == 'wiki':
    t1 = 15
    t2 = 20
    min_epoch_t = 120
elif dataset == 'pubmed':
    t1 = 5
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

    network = ReDGAE(adj=adj_norm, num_neurons=num_neurons, num_features=num_features,
                     embedding_size=embedding_size, nClusters=nClusters, activation="ReLU", alpha=alpha,
                     gamma=gamma)

    if i == 0:
        network.pretrain(adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam",
                         epochs=epochs_pretrain, lr=lr_pretrain, save_path=save_path, dataset=dataset)

    mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = network.train(adj_norm, features, adj, adj_label, labels, n_outliers, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_cluster, lr=lr_cluster, beta1=beta1, beta2=beta2, save_path=save_path, dataset=dataset, t1=t1, t2=t2, min_epoch_t=min_epoch_t)

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
