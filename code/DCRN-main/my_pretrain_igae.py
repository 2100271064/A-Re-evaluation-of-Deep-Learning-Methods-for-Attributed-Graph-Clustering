'''
Author: xinying Lai
Date: 2022-09-13 15:59:21
LastEditTime: 2022-09-13 20:01:42
Description: Do not edit
'''
# 训练IGAE子网络 epoch=30
import opt
from DCRN import GNNLayer, IGAE_encoder, IGAE_decoder
from utils_mod import *
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os


class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj, _, _ = self.encoder(x, adj)
        z_hat, z_hat_adj, _, _ = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat


# GPU是否开启
if torch.cuda.is_available():
    print("Available GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cuda")


dataset_name = opt.args.name
root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset_name)
x = np.loadtxt(root_path + '/feature.txt', dtype=float)
vertex_arr = np.array(pd.read_table(root_path + "/node.txt", header=None)[0])
x = x[vertex_arr]

# TODO 原来是打开的
# # 其他数据集均n_input=100(参照DFCN)
# if dataset_name == "dblp":
#     n_input = 50
# else:
#     n_input = 100
if (opt.args.name == "dblp"):
    opt.args.n_components = 256
pca = PCA(n_components=opt.args.n_components, svd_solver='full')
X_pca = pca.fit_transform(x)
X = numpy_to_torch(X_pca).to(device)

adj = np.loadtxt(root_path + '/adj.txt', dtype=int)
adj = adj[vertex_arr]
adj = adj[:, vertex_arr]
# 报错
# adj = numpy_to_torch(adj).to(device)
# 按照DFCN中修改
adj = sp.coo_matrix(adj)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

# 预训练IGAE模型
def pretrain_igae(model, X, adj, y, n_outliers, lr, n_clusters, device):

    # 检查是否有用gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("当前设备索引：", torch.cuda.current_device())
    else:
        print("no cude", device.type)

    print("\n", model)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(30):

        z_igae, z_hat, adj_hat = model(X, adj)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, X))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_igae = loss_w + 0.1 * loss_a
        loss = loss_igae
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_igae.data.cpu().numpy())
        eva(y, n_outliers, kmeans.labels_, epoch)

        torch.save(model.state_dict(), '/home/laixy/AHG/data/DCRN-main/model_pre/{0}_igae.pkl'.format(dataset_name))

# citeseer、acm、dblp、pubmed按照论文设置的学习率, 其他的默认1e-5
lr = 1e-5
if opt.args.name == 'acm':
    lr = 5e-5
elif opt.args.name == 'dblp':
    lr = 1e-4


y = np.loadtxt(root_path + '/label.txt', dtype=int)

model = IGAE(
    gae_n_enc_1=opt.args.gae_n_enc_1,
    gae_n_enc_2=opt.args.gae_n_enc_2,
    gae_n_enc_3=opt.args.gae_n_enc_3,
    gae_n_dec_1=opt.args.gae_n_dec_1,
    gae_n_dec_2=opt.args.gae_n_dec_2,
    gae_n_dec_3=opt.args.gae_n_dec_3,
    n_input=opt.args.n_components
    ).to(device)

[n_clusters, n_outliers] = cal_n_cluster(dataset_name)
pretrain_igae(model, X, adj, y, n_outliers, lr=lr, n_clusters=n_clusters, device=device)