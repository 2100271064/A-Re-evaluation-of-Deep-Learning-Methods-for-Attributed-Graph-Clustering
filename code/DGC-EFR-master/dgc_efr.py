from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='pubmed')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--k',type=int, default=5)
# TODO 原来是200
parser.add_argument('--epochs', type=int, default=400)
# loss
parser.add_argument('--lambda1', type=float, default=1)
parser.add_argument('--n_clusters', default=6, type=int)
parser.add_argument('--update_interval', default=1, type=int)
parser.add_argument('--hidden1_dim', default=1024, type=int)
parser.add_argument('--hidden2_dim', default=256, type=int)
parser.add_argument('--hidden3_dim', default=16, type=int)
parser.add_argument('--n_z', default=10, type=int)
parser.add_argument('--weight_decay', type=int, default=5e-3)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--n_components', type=int, default=500)
# ?
# parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
# ?
# parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
# ?
# parser.add_argument('--save', type=str, default='')
args = parser.parse_args()

# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys
import time
import glob
import logging
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils_mod import *
from GNN import GraphAttentionLayer
from evaluation import eva
from sklearn.decomposition import PCA

# The basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        # degree
        self.v = v

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

# graph autoencoder
class GAE(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
        super(GAE, self).__init__()
        # encoder
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden1_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden1_size, hidden2_size, alpha)
        self.conv3 = GraphAttentionLayer(hidden2_size, embedding_size, alpha)

    def forward(self, x, adj, M, enc_feat1, enc_feat2, enc_feat3):
        sigma = 0.1
        h = self.conv1(x, adj, M)
        h = self.conv2((1-sigma)*h + sigma*enc_feat1, adj, M)
        h = self.conv3((1-sigma)*h + sigma*enc_feat2, adj, M)
        z = F.normalize((1 -sigma)*h + sigma*enc_feat3, p=2, dim=1)
        # decoder
        A_pred = dot_product_decode(z)
        return A_pred, z


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class DGCEFR(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha, num_clusters, n_z, v=1):
        super(DGCEFR, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.v = v

        # pre-trained ae
        self.pre_ae =  AE(hidden1_size, hidden2_size, embedding_size, embedding_size, hidden2_size, hidden1_size, num_features, n_z = n_z)
        # TODO
        self.pre_ae.load_state_dict(torch.load(args.preae_path, map_location='cpu'))
        # pre-trained gae
        self.pre_gae = GAE(num_features, hidden1_size, hidden2_size, embedding_size, alpha)
        # TODO
        self.pre_gae.load_state_dict(torch.load(args.pregae_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        x_bar, enc_h1, enc_h2, enc_h3, z = self.pre_ae(x)
        A_pred, z = self.pre_gae(x, adj, M, enc_h1, enc_h2, enc_h3)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, z, q, x_bar


def train(dataset, args):
    # load initialized model
    model = DGCEFR(num_features=args.input_dim, hidden1_size=args.hidden1_dim, hidden2_size=args.hidden2_dim,
                  embedding_size=args.hidden3_dim, alpha=args.alpha, num_clusters=args.n_clusters, n_z=args.n_z).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # adjacent matrix
    adj, adj_label = load_graph(args.name)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = sklearn.preprocessing.normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y

    with torch.no_grad():
         _, z,_, _ = model(data, adj, M)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, nmi, ari, f1 = eva(y, args.n_outliers, y_pred, 'pae')
    # logging.info('epoch %d acc %.4f nmi %.4f ari %.4f f1 %.4f', 0, acc, nmi, ari, f1)
    # acc_best, nmi_best, ari_best, f1_best, epoch_best = 0.0, 0.0, 0.0, 0.0, 0
    listacc = []
    listnmi = []
    listari = []
    listf1 = []
    for epoch in range(args.epochs):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, tmp_q, x_bar = model(data, adj, M)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P

            eva(y, args.n_outliers, res2, str(epoch) + 'P')
            # logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' P', acc, nmi, ari, f1)
            acc, nmi, ari, f1 = eva(y, args.n_outliers, res1, str(epoch) + 'Q')
            # logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' Q', acc, nmi, ari, f1)
            if acc != -1:
                listacc.append(acc)
                listnmi.append(nmi)
                listari.append(ari)
                listf1.append(f1)

        A_pred, z, q, x_bar = model(data, adj, M)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss_gae = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        re_loss_ae = F.mse_loss(x_bar, data)
        # loss function with 3 parts
        # loss = args.lambda1 * kl_loss + 1.0*re_loss_gae + 10*re_loss_ae
        loss = kl_loss + re_loss_gae + re_loss_ae

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    # print(epoch_best, ':acc_best {:.4f}'.format(acc_best), ', nmi {:.4f}'.format(nmi_best), ', ari {:.4f}'.format(ari_best),
    #         ', f1 {:.4f}'.format(f1_best))
    return np.mean(listacc), np.mean(listnmi), np.mean(listari), np.mean(listf1), len(listacc)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='train',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--name', type=str, default='acm')
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--k',type=int, default=5)
    # parser.add_argument('--epochs', type=int, default=200)
    # # loss
    # parser.add_argument('--lambda1', type=float, default=1)
    # parser.add_argument('--n_clusters', default=6, type=int)
    # parser.add_argument('--update_interval', default=1, type=int)
    # parser.add_argument('--hidden1_dim', default=1024, type=int)
    # parser.add_argument('--hidden2_dim', default=256, type=int)
    # parser.add_argument('--hidden3_dim', default=16, type=int)
    # parser.add_argument('--n_z', default=10, type=int)
    # parser.add_argument('--weight_decay', type=int, default=5e-3)
    # parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    # # ?
    # # parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    # # ?
    # # parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    # # ?
    # # parser.add_argument('--save', type=str, default='')
    # args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # args.save = './exp/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    # create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    #
    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    # load data
    dataset = load_data(args.name)

    # TODO 特征工程
    if (args.name == "dblp"):
        args.n_components = 256
    pca = PCA(n_components=args.n_components, svd_solver='full')
    dataset.x = pca.fit_transform(dataset.x)
    print("PCA后,feature.shape=", dataset.x.shape)

    args.input_dim = dataset.x.shape[1]
    [args.n_clusters, args.n_outliers] = cal_n_cluster(args.name)
    args.k = None

    # some configurations for datasets
    # TODO 原来是打开的
    # if args.name == 'acm' or args.name == 'wiki':
    #     args.epochs = 200
    #     args.lr = 5e-3
    #
    # if args.name == 'dblp' or args.name == 'pubmed':
    #     args.lr = 0.01
    #     args.epochs = 200
    #
    # if args.name == 'cora':
    #     args.epochs = 50
    #     args.lr = 0.001
    #
    # if args.name == 'citeseer':
    #     args.epochs = 100
    #     args.lr = 0.001

    args.lr = 0.0001

    # load pre-trained models
    root_path = r"/home/laixy/AHG/data/DGC-EFR-master"
    args.pregae_path = root_path + r'/model_pre/{}_gae.pkl'.format(args.name)
    args.preae_path = root_path + r'/model_pre/{}_ae.pkl'.format(args.name)
    print(args)

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    valid_epoch_num_list = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = train(dataset, args)

        final_acc.append(mean_acc)
        final_nmi.append(mean_nmi)
        final_ari.append(mean_ari)
        final_f1.append(mean_f1)
        valid_epoch_num_list.append(valid_epoch_num)

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)
    print("{} epoch × {}, 有效的epoch数：".format(args.epochs, run_round), valid_epoch_num_list)

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
