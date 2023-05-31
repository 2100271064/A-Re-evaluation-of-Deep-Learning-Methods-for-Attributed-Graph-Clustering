import opt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


class GCN_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(GCN_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z1 = self.gnn_1(x, adj, active=True)
        z2 = self.gnn_2(z1, adj, active=True)
        z = self.gnn_3(z2, adj, active=False)
        z_adj = self.s(torch.mm(z, z.t()))
        return z, z_adj


class GCN_decoder(nn.Module):

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(GCN_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z, adj):
        z = self.gnn_4(z, adj, active=True)
        z = self.gnn_5(z, adj, active=True)
        z_hat = self.gnn_6(z, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class GCN(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, n_clusters, v):
        super(GCN, self).__init__()
        self.encoder = GCN_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.enc_cluster = GNNLayer(gae_n_enc_3, n_clusters)

        self.decoder = GCN_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

        self.v = v
        self.n_clusters = n_clusters
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, gae_n_enc_3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj):
        z, z_adj = self.encoder(x, adj)
        z_cluster = self.enc_cluster(z, adj)
        z_hat, z_hat_adj = self.decoder(z, adj)
        # 说如果要用这种A的解码方式，是否要sigmoid处理一下
        adj_hat = z_adj + z_hat_adj

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z_hat, z_adj, z, z_cluster, q
