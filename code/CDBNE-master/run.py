import argparse

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--weight_decay', type=int, default=5e-3)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--alpha1', type=float, default=1, help="p and q")
parser.add_argument('--name', type=str, default='citeseer')
# parser.add_argument('--epoch', type=int, default=30)
# TODO 原来是500
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_clusters', default=12, type=int)
parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
parser.add_argument('--middle_size', default=256, type=int)
parser.add_argument('--representation_size', default=16, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--n_components', type=int, default=500)

args = parser.parse_args()

# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import matplotlib.pyplot as plt
# from torch_geometric.datasets import Planetoid
from sklearn.manifold import TSNE
import pandas as pd
from torch.nn import init
import matplotlib.pyplot as plt


from utils_mod import *
from model import GATE, Modula
from evaluation import eva
from sklearn.decomposition import PCA


# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
#
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此处还可以传入其他层
#     )
# print(net)
# print(net[0])
# from visiuation import plot_embedding


# def maxmetr(acc, nmi, ari, f1):
#     print(f"max :acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
#     return


class CDBNE(nn.Module):
    def __init__(self, num_features, middle_size, representation_size, alpha, clusters_number, v=1):
        super(CDBNE, self).__init__()
        self.clusters_number = clusters_number
        self.v = v

        # get model
        self.gate = GATE(num_features, middle_size, representation_size, alpha)
        # TODO 预训练加载
        self.gate.load_state_dict(torch.load(args.pathes, map_location='cpu'))

        # cluster layer   # cluster layer，簇头embed
        self.cluster_layer = Parameter(torch.Tensor(clusters_number, representation_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        # 得到reconstruct的邻接和[N, feat_size]的节点embedding Z
        A_pred, z_embedding = self.gate(x, adj, M)
        q = self.modularity(z_embedding)
        return A_pred, z_embedding, q

    def modularity(self, z_embedding):
        dist = torch.sum(torch.pow(z_embedding.unsqueeze(1) - self.cluster_layer, 2), 2)
        q = 1.0 / (1.0 + dist / self.v) ** ((self.v + 1.0) / 2.0)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_fenbu(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def trainer(adj_normalized, adj_label, x, y):
    model = CDBNE(num_features=args.input_dim, middle_size=args.middle_size,
                  representation_size=args.representation_size, alpha=args.alpha, clusters_number=args.n_clusters).to(
        args.device)
    # print(model)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer1 = optim.SGD(net.parameters(), lr=0.03)
    # data process
    # shujuji = utils.data_preprocessing(shujuji)
    # adj = shujuji.adj.to(device)
    # adj_label = shujuji.adj_label.to(device)
    adj = adj_normalized.to_dense().to(args.device)
    adj_label = torch.Tensor(adj_label).to(args.device)
    # if args.n_outliers > 0:
    #     adj_label = adj_label[:-args.n_outliers, :-args.n_outliers]
    M = get_M(adj).to(args.device)

    # data label
    # data = torch.Tensor(shujuji.x).to(device)
    # y = shujuji.y.cpu().numpy()
    data = torch.Tensor(x).to(args.device)
    # y的尺寸为2708 [3 4 4 ... 3 3 3]

    with torch.no_grad():
        # 相当epoch的模型做一eval
        _, z_embedding = model.gate(data, adj, M)

    # get kmeans and pre result
    # 这里是用pre结果来初始化kmean中心
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    kmeans.fit(z_embedding.data.cpu())
    # vis.plot_embedding(z_embedding.data.cpu())
    y_pred = kmeans.fit_predict(z_embedding.data.cpu().numpy())  # 得到label
    # y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    # plt.scatter(z_embedding.data.cpu().numpy()[:, 0], z_embedding.data.cpu().numpy()[:, 1], c=y_pred)
    # plt.show()
    # y_pred = DBSCAN().fit_predict(z_embedding.data.cpu().numpy())
    # eva(y, y_pred, 'pre')
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    # print(kmeans.cluster_centers_)
    # ADj = np.array(utils.data_preprocessing(shujuji).adj)  # 得到正常邻接矩阵
    listacc = []
    listnmi = []
    listari = []
    listf1 = []

    # i = 1
    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z_embedding, Q = model(data, adj, M)
            q = Q.detach().data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(y, args.n_outliers, q, epoch)
            if acc != -1:
                listacc.append(acc)
                listnmi.append(nmi)
                listari.append(ari)
                listf1.append(f1)

        A_pred, z_embedding, q = model(data, adj, M)
        # 输入 2708*16
        # z_embeddin = TSNE(n_components=2).fit_transform(z_embedding.data.cpu().numpy())
        # '''分隔符'''  # 这里是聚类可视化
        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # # plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=y_pred, s=20)
        # color_set = ('gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue')
        # color_list = [color_set[int(label)] for label in y_pred]
        # plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
        # plt.show()
        '''分隔符'''
        # fig = plot_embedding(z_embeddin, y, 'julei')
        # 显示图像
        # plt.show()
        p = target_fenbu(Q.detach())
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.contiguous().view(-1), adj_label.contiguous().view(-1))
        MU_loss = Modula(adj.detach().cpu().numpy(), A_pred.detach().cpu().numpy())

        # loss = 0.001 * kl_loss + re_loss - 0.1 * MU_loss
        loss = 1 * kl_loss + re_loss - 0.1 * MU_loss
        print('{} loss: {}'.format(epoch, loss))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer1.step()
    # print(f"max :acc {max(listacc):.4f}, nmi {max(listnmi):.4f}, ari {max(listari):.4f}, f1 {max(listf1):.4f}")
    return np.mean(listacc), np.mean(listnmi), np.mean(listari), np.mean(listf1), len(listacc)


# def GetPrearr(x, num_cluster):
#     matrix = np.zeros((len(x), num_cluster))
#     for i in range(len(x)):
#         for j in range(num_cluster):
#             matrix[i][j] = 0
#             matrix[i][x[i]] = 1
#     return matrix

if __name__ == "__main__":

    args.cuda = torch.cuda.is_available()
    print(args.cuda)
    setup(args)

    # device = torch.device("cuda" if args.cuda else "cpu")
    # datasets = utils.get_dataset(args.name)
    # shujuji = datasets[0]

    # if args.name == 'Cora':
    #     args.lr = 0.0001
    #     args.k = None
    #     args.n_clusters = 7
    #     args.epoch = 1
    #     args.v = 1
    # else:
    #     args.k = None

    args.pathes = "/home/laixy/AHG/data/CDBNE-master/model_pre/{}.pkl".format(args.name)
    # args.input_dim = shujuji.num_features

    print(args)
    # trainer(shujuji)
    adj, adj_normalized, features, label = load_data(args.name)

    # TODO 特征工程
    if (args.name == "dblp"):
        args.n_components = 256
    pca = PCA(n_components=args.n_components, svd_solver='full')
    features = pca.fit_transform(features)
    print("PCA后,feature.shape=", features.shape)

    args.input_dim = features.shape[1]

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    valid_epoch_num_list = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = trainer(adj_normalized, adj, features, label)

        final_acc.append(mean_acc)
        final_nmi.append(mean_nmi)
        final_ari.append(mean_ari)
        final_f1.append(mean_f1)
        valid_epoch_num_list.append(valid_epoch_num)

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)
    print("{} epoch × {}, 有效的epoch数：".format(args.max_epoch, run_round), valid_epoch_num_list)

    value = np.mean(acc_arr)
    var = np.var(acc_arr)
    print('final_acc: {}, fianl_var_acc: {}'.format(value, var))
    print('final_acc: {:.4f}, fianl_var_acc: {:.2f}'.format(value, var))

    value = np.mean(nmi_arr)
    var = np.var(nmi_arr)
    print('final_nmi: {}, final_var_nmi: {}'.format(value, var))
    print('final_nmi: {:.4f}, final_var_nmi: {:.2f}'.format(value, var))

    value = np.mean(ari_arr)
    var = np.var(ari_arr)
    print('final_ari: {}, final_var_ari: {}'.format(value, var))
    print('final_ari: {:.4f}, final_var_ari: {:.2f}'.format(value, var))

    value = np.mean(f1_arr)
    var = np.var(f1_arr)
    print('final_f1: {}, final_var_f1: {}'.format(value, var))
    print('final_f1: {:.4f}, final_var_f1: {:.2f}'.format(value, var))