'''
Author: xinying Lai
Date: 2022-09-13 20:03:25
LastEditTime: 2022-09-13 20:20:18
Description: Do not edit
'''
# 训练IGAE子网络 epoch=30
import opt
from DCRN import *
from utils_mod import *
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# GPU是否开启
if opt.args.cuda and torch.cuda.is_available():
    print("Available GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# 将超参数设置好
setup()

# data pre-precessing: X, y, A, A_norm, Ad
X, y, A = load_graph_data(opt.args.name, show_details=False)
A_norm = normalize_adj(A, self_loop=True, symmetry=False)
Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)

# to torch tensor
X = numpy_to_torch(X).to(opt.args.device)
A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
Ad = numpy_to_torch(Ad).to(opt.args.device)

# 预训练DCRN模型
def pretrain_DCRN(model, X, y, n_outliers, A, A_norm, Ad, n_clusters):

    # 检查是否有用gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("当前设备索引：", torch.cuda.current_device())
    else:
        print("no cude", device.type)

    print("\n", model)

    # calculate embedding similarity
    with torch.no_grad():
        _, _, _, sim, _, _, _, Z, _, _ = model(X, A_norm, X, A_norm)
    # calculate cluster centers
    _, _, _, _, centers = clustering(Z, y, n_outliers, n_clusters)
    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)
    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    Am = remove_edge(A, sim, remove_rate=0.1)

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    for epoch in range(100):
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        # split batch
        index = list(range(len(X_tilde1)))
        split_list = data_split(index, 2000)

        X_tilde1 = X_tilde1.to(opt.args.device)
        X_tilde2 = X_tilde2.to(opt.args.device)
        model = model.to(opt.args.device)

        print("共有{}个batch".format(len(split_list)))

        batch_count = 0
        # mini-batch
        for batch in split_list:

            batch_count = batch_count + 1

            X_tilde1_batch = X_tilde1[batch]
            X_tilde2_batch = X_tilde2[batch]

            Ad_batch = Ad[batch, :][:, batch]
            Ad_batch = Ad_batch.to(opt.args.device)

            Am_batch = Am[batch, :][:, batch]
            Am_batch = Am_batch.to(opt.args.device)

            # input & output
            X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1_batch, Ad_batch,
                                                                                     X_tilde2_batch, Am_batch)

            # calculate loss: L_{DICR}, L_{REC} and L_{KL}
            L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all)
            L_REC = reconstruction_loss(X[batch], A_norm.to_dense()[batch, :][:, batch], X_hat, Z_hat, A_hat)
            L_KL = distribution_loss(Q, target_distribution(Q[0].data))
            loss = L_DICR + L_REC + opt.args.lambda_value * L_KL

            # optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            print("当前epoch={}, 当前batch={}, loss={}".format(epoch, batch_count, loss))

        if epoch % 20 == 0:
            model_cpu = model.cpu()
            X_tilde1 = X_tilde1.cpu()
            X_tilde2 = X_tilde2.cpu()
            Ad = Ad.cpu()
            Am = Am.cpu()

            with torch.no_grad():
                X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model_cpu(X_tilde1, Ad, X_tilde2, Am)

            acc, nmi, ari, f1, _ = clustering(Z, y, n_outliers, n_clusters)

            Ad = Ad.to(opt.args.device)
            Am = Am.to(opt.args.device)

        # 数据仍然保存在DCRN中
        torch.save(model.state_dict(), '/home/laixy/AHG/data/DCRN-main/model_pre/{0}_DCRN.pkl'.format(opt.args.name))

root_path = "/home/laixy/AHG/dataset/{}".format(opt.args.name)

y = np.loadtxt(root_path + '/label.txt', dtype=int)
[n_clusters, n_outliers] = cal_n_cluster(opt.args.name)

# Dual Correlation Reduction Network
model = DCRN(n_node=X.shape[0]).to(opt.args.device)

pretrain_DCRN(model, X, y, n_outliers, A, A_norm, Ad, n_clusters)