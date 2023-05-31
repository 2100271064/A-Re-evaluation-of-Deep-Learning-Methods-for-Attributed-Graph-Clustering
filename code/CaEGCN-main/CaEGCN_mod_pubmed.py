from __future__ import print_function, division
import argparse
parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# TODO
parser.add_argument('--name', type=str, default='pubmed')
parser.add_argument('--k', type=int, default=3)
# TODO 原来是5e-4
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_clusters', default=3, type=int)
parser.add_argument('--n_z', default=10, type=int)
parser.add_argument('--pretrain_path', type=str, default='pkl')
# TODOchou'kon
# parser.add_argument('--version', type=str, default='max_cluster_num_5')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--cuda', type=bool, default=True)
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
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b = 1
        t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)



#   reconstruct graph
def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    # print("dot_product_decode:A_pred", A_pred, torch.max(A_pred), torch.min(A_pred))
    return A_pred

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # (wiki)4973 -> 500
        self.enc_1 = Linear(n_input, n_enc_1)
        # 500 -> 500
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        # 500 -> 2000
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # 2000 -> 10
        self.z_layer = Linear(n_enc_3, n_z)

        # 10 -> 2000
        self.dec_1 = Linear(n_z, n_dec_1)
        # 2000 -> 500
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        # 500 -> 500
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        # 500 -> (wiki)4973
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

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h1, dec_h2, dec_h3  # 将encoder和decoder都返回


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

        # # GCN for inter information
        # self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        # self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)
        # self.gnn_5 = GNNLayer(n_z, n_clusters)

        # (wiki)4973 -> 500
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)#
        # 500 -> 10
        self.gnn_3 = GNNLayer(n_enc_1, n_z)  #
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)#
        # 10 -> (wiki)17
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        # (wiki)17 -> 2000
        self.gnn_6 = GNNLayer(n_clusters, n_dec_1)
        # 2000 -> 500
        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        # self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        # 500 -> (wiki)4973
        self.gnn_9 = GNNLayer(n_dec_2, n_input)
        # self.attn1 = SelfAttentionWide(n_enc_1)
        # self.attn2 = SelfAttentionWide(n_enc_2)
        # √
        self.attn3 = SelfAttentionWide(n_enc_1)
        # self.attn4 = SelfAttentionWide(n_z)
        # √
        self.attn5 = SelfAttentionWide(n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_1, dec_2, dec_3 = self.ae(x)  # 增加了decoder的输出
        # print(x.shape)
        # print("ae:x_bar", x_bar)
        # print("ae:z", z)

        # GCN Module
        h = self.gnn_1(x, adj)  # 相加

        h = self.attn3((h + tra2))
        # 上面返回三维，第一维是1，去掉它，重新变成二维
        h = h.squeeze(0)
        h = self.gnn_3(h, adj)

        # print(h.shape)
        h1 = self.attn5((h+z))
        h1 = h1.squeeze(0)
        h1 = self.gnn_5(h1, adj, active=False)
        # h1.unsqueeze(0)
        # print(h1.shape)
        # h1 = self.attn5(h1)
        # h1 = h1.squeeze(0)
        # print(h1.shape)

        # 解决exp溢出问题
        if args.name == "wiki":
            predict = F.log_softmax(h1, dim=1)
        #     tmp_h1 = h1
        #     for row_index in range(tmp_h1.shape[0]):
        #         tmp_h1[row_index] = tmp_h1[row_index] - tmp_h1[row_index].max()
        #         predict = F.softmax(tmp_h1, dim=1)
        else:
            predict = F.softmax(h1, dim=1)
        # predict = F.softmax(h1, dim=1)
        # print("forward:predict", predict)
        h = self.gnn_6(h1, adj)
        h = self.gnn_7(h + dec_1, adj)
        # h = self.gnn_8(h + dec_2, adj)
        h = self.gnn_9(h + dec_3, adj)
        # print("forward:h", h)
        A_pred = dot_product_decode(h)
        # print("forward:A_pred", A_pred)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, h , A_pred ,h1


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, n_outliers):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0)
    print(model)
    
    # 检查是否有用gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    else:
        print("no cude", device.type)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name)

    data = torch.Tensor(dataset.x)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z, _, _, _ = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, max_iter=1000)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, n_outliers, y_pred, 'pae')

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    # TODO 原来是700
    for epoch in range(400):
        if epoch % 20 == 0:

            model = model.cpu()
            data = data.cpu()
            adj = adj.cpu()

            with torch.no_grad():
                _, tmp_q, pred, _, h , A_pred, h1 = model(data, adj)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)  # 计算p是一种自增强

            res1 = tmp_q.numpy().argmax(1)  # Q
            # print(type(res1))
            res2 = pred.data.numpy().argmax(1)  # Z
            # print(type(res2))
            res3 = p.data.numpy().argmax(1)  # P
            eva(y, n_outliers, res1, str(epoch) + 'Q')
            acc, nmi, ari, f1 = eva(y, n_outliers, res2, str(epoch) + 'Z')
            if acc != -1:
                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)
                f1_list.append(f1)
            eva(y, n_outliers, res3, str(epoch) + 'P')

        # split batch
        index = list(range(len(data)))
        split_list = data_split(index, 2000)

        model = model.to(device)
        data = data.to(device)

        print("共有{}个batch".format(len(split_list)))

        batch_count = 0
        # mini-batch
        for batch in split_list:

            batch_count = batch_count + 1

            data_batch = data[batch]
            adj_batch = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj.to_dense()[batch, :][:, batch]))
            adj_batch = adj_batch.to(device)

            x_bar, q, pred, _, h, A_pred, h1 = model(data_batch, adj_batch)

            tmp_q = q.data
            p = target_distribution(tmp_q)

            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            # print("当前epoch={}, kl_loss={}".format(epoch, kl_loss))
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')  # 与概率p比？
            # print("当前epoch={}, ce_loss={}".format(epoch, ce_loss))
            re_loss = F.mse_loss(x_bar, data_batch)
            # print("当前epoch={}, re_loss={}".format(epoch, re_loss))
            graph_loss = F.mse_loss(h, data_batch)
            # print("当前epoch={}, graph_loss={}".format(epoch, graph_loss))
            re_graphloss = F.binary_cross_entropy(A_pred.view(-1), adj_batch.to_dense().view(-1))
            # print("当前epoch={}, re_graphloss={}".format(epoch, re_graphloss))

            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss+0.01 * re_graphloss + 0.01 * graph_loss
            print("当前epoch={}, 当前batch={}, loss={}".format(epoch, batch_count, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print("Optimization Finished!")
    acc_arr = np.array(acc_list)
    nmi_arr = np.array(nmi_list)
    ari_arr = np.array(ari_list)
    f1_arr = np.array(f1_list)

    return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr), len(acc_arr)
     


from warnings import simplefilter

if __name__ == "__main__":

    args.cuda = args.cuda and torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)

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

    # TODO 原来是打开的
    # if args.name == 'dblp':
    #     args.lr = 1e-4

    # TODO 原来是打开的
    # if args.name == 'citeseer':
    #     args.lr = 5e-5

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
        print("round_{0}:\nmean_acc {1}\nmean_nmi {2}\nmean_ari {3}\nmean_f1 {4}\n有效epoch数:{5}".format(i, mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num))

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)
    print("700 epoch × 10, 有效的epoch数：", valid_epoch_num_list)

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