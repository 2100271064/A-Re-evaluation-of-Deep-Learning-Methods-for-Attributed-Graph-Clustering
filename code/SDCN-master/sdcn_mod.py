from __future__ import print_function, division
import argparse
parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# TODO
parser.add_argument('--name', type=str, default='wiki')
parser.add_argument('--k', type=int, default=3)
# TODO 原来是1e-3
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_clusters', default=3, type=int)
parser.add_argument('--n_z', default=10, type=int)
parser.add_argument('--pretrain_path', type=str, default='pkl')
# TODO
# parser.add_argument('--version', type=str, default='max_cluster_num_5')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_components', type=int, default=500)
args = parser.parse_args()

import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils_mod import *
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from sklearn.decomposition import PCA


# torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, n_outliers):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, n_outliers, y_pred, 'pae')

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    # TODO 原来是200
    for epoch in range(400):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, n_outliers, res1, str(epoch) + 'Q')
            acc, nmi, ari, f1 = eva(y, n_outliers, res2, str(epoch) + 'Z')
            if acc != -1:
                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)
                f1_list.append(f1)
            eva(y, n_outliers, res3, str(epoch) + 'P')

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Optimization Finished!")
    acc_arr = np.array(acc_list)
    nmi_arr = np.array(nmi_list)
    ari_arr = np.array(ari_list)
    f1_arr = np.array(f1_list)

    return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr), len(acc_arr)


if __name__ == "__main__":

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # 预训练模型的路径
    args.pretrain_path = '/home/laixy/AHG/data/SDCN-master/model_pre/{0}.pkl'.format(args.name)
    dataset = load_data(args.name)

    # TODO 特征工程
    # if (args.name == "dblp"):
    #     args.n_components = 256
    # pca = PCA(n_components=args.n_components, svd_solver='full')
    # dataset.x = pca.fit_transform(dataset.x)

    [args.n_clusters, n_outliers] = cal_n_cluster(args.name)
    args.n_input = dataset.x.shape[1]
    args.k = None


    # if args.name == 'acm':
    #     pass

    # if args.name == 'dblp':
    #     pass

    # TODO 原来是打开的
    # if args.name == 'citeseer':
    #     args.lr = 1e-4

    # if args.name == 'cora':
    #     pass
    
    # if args.name == 'wiki':
    #     pass
    
    # if args.name == 'pubmed':
    #     pass

    print(args)

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    valid_epoch_num_list = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))
        mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = train_sdcn(dataset, n_outliers)
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
