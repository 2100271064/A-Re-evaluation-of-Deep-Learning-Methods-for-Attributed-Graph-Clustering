'''
Author: xinying Lai
Date: 2022-09-07 15:00:14
LastEditTime: 2022-09-13 18:30:38
Description: Do not edit
'''
import random

import logging
from torch_scatter import scatter
import opt as opt
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
import torch
import numpy as np
from GAE import IGAE,IGAE_encoder

from utils_mod import setup_seed, cal_n_cluster, normalize_adj, sparse_mx_to_torch_sparse_tensor
from train_mod import Train_gae
from sklearn.decomposition import PCA
from load_data import *
import networkx as nx
from networkx.readwrite import json_graph
import json

import warnings

from view_learner import ViewLearner

warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
setup_seed(np.random.randint(1000))


import pandas as pd

# pandas设置显示最大列数
pd.set_option('display.max_columns', None)
# pandas设置显示最大行数
pd.set_option('display.max_rows', None)

print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

# 保存的特征
if opt.args.name != "uk" and opt.args.name != "us":
    opt.args.data_path = '/home/laixy/AHG/dataset/{}/feature.txt'.format(opt.args.name)
else:
    opt.args.data_path = '/home/laixy/AHG/dataset/{}/feat192.npy'.format(opt.args.name)
# 保存的标签
opt.args.label_path = '/home/laixy/AHG/dataset/{}/label.txt'.format(opt.args.name)
# 保存的KNN邻居
# opt.args.graph_k_save_path = '/home/laixy/AHG/AGC-DRR-main/graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
# 保存的图(自己构造graph)
# opt.args.graph_save_path = '/home/laixy/AHG/dataset/{}/graph.txt'.format(opt.args.name)
# 保存的模型
# opt.args.model_save_path = '/home/laixy/AHG/AGC-DRR-main/model/model_save_gae/{}_gae.pkl'.format(opt.args.name)


print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

if opt.args.name != "uk" and opt.args.name != "us":
    x = np.loadtxt(opt.args.data_path, dtype=float)
else:
    x = np.load(opt.args.data_path)

vertex_arr = np.loadtxt('/home/laixy/AHG/dataset/{}/node.txt'.format(opt.args.name), dtype=int)
x = x[vertex_arr]

y = np.loadtxt(opt.args.label_path, dtype=int)
print("feature.shape=", x.shape, "feature.type=", type(x))
print("label.shape=", y.shape, "label.type=", type(y))

if opt.args.name != "uk" and opt.args.name != "us":
    adj = np.loadtxt('/home/laixy/AHG/dataset/{}/adj.txt'.format(opt.args.name), dtype=int)

    adj = adj[vertex_arr]
    adj = adj[:, vertex_arr]

    # 自己构造(顶点重新编号)
    edge_list = []
    for i in range(adj.shape[0]):
        j_list = np.where(adj[i] == 1)[0]
        for j in j_list:
            edge_list.append([np.int32(i), np.int32(j)])
    edge_index1 = np.array(edge_list)
    print("转置前:", edge_index1.shape)
    # 转置
    edge_index1 = edge_index1.transpose()
    print("转置后:", edge_index1.shape)

    # 归一化邻接矩阵
    adj = normalize_adj(adj)
    adj = torch.Tensor(adj).to(device)

    # TODO 特征工程
    # if(opt.args.name == "dblp"):
    #         opt.args.n_components = 256
    # pca1 = PCA(n_components=opt.args.n_components, svd_solver='full')
    # x1 = pca1.fit_transform(x)
    dataset = LoadDataset(x)
    data = torch.Tensor(dataset.x).to(device)
    # print("PCA后,feature.shape=", data.shape)
else:
    nx_graph = json_graph.node_link_graph(json.load(open('/home/laixy/AHG/dataset/{}/(renumberated)graph.json'.format(opt.args.name))))
    adj = nx.to_scipy_sparse_matrix(nx_graph)

    edge_list = list(nx_graph.edges)
    edge_index1 = np.array(edge_list)
    print("转置前:", edge_index1.shape)
    # 转置
    edge_index1 = edge_index1.transpose()
    print("转置后:", edge_index1.shape)

    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    data = torch.Tensor(x).to(device)


print("adj.shape=", adj.shape, "adj.type=", type(adj))


[opt.args.n_clusters, n_outliers] = cal_n_cluster(opt.args.name)


run_round = 10
final_acc = []
final_nmi = []
final_ari = []
final_f1 = []
valid_epoch_num_list = []
for i in range(run_round):
    print('----------------------round_{0}-----------------------------'.format(i))

    # N2
    model_gae = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_enc_3=opt.args.gae_n_enc_3,
            # 特征工程后的特征新维度
            n_input=data.shape[1]
        ).to(device)

    # N1
    view_learner = ViewLearner(
            IGAE_encoder(gae_n_enc_1=opt.args.gae_n_enc_1,
                        gae_n_enc_2=opt.args.gae_n_enc_2,
                        gae_n_enc_3=opt.args.gae_n_enc_3,
                        n_input=data.shape[1]),
        ).to(device)

    mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = Train_gae(model_gae,view_learner, data, adj, y, n_outliers, edge_index1, device)
    final_acc.append(mean_acc)
    final_nmi.append(mean_nmi)
    final_ari.append(mean_ari)
    final_f1.append(mean_f1)
    valid_epoch_num_list.append(valid_epoch_num)

acc_arr = np.array(final_acc)
nmi_arr = np.array(final_nmi)
ari_arr = np.array(final_ari)
f1_arr = np.array(final_f1)
print("200 epoch × 10, 有效的epoch数：", valid_epoch_num_list)

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


