import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import opt
from AE import *
from GCN import *
from ATT_AE2 import *

class Merge(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_dec_1, n_dec_2, n_dec_3, n_z,
                 n_input, n_clusters, alpha, v=1.0):
        super(Merge, self).__init__()

        self.ae = AE(
            ae_n_enc_1=n_enc_1,
            ae_n_enc_2=n_enc_2,
            ae_n_enc_3=n_enc_3,
            ae_n_dec_1=n_dec_1,
            ae_n_dec_2=n_dec_2,
            ae_n_dec_3=n_dec_3,
            n_z=n_z,
            n_input=n_input,
            n_clusters=n_clusters)
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gcn_n_enc_1 = GNNLayer(n_input, n_enc_1)
        self.gcn_n_enc_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gcn_n_enc_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gcn_n_enc_4 = GNNLayer(n_enc_3, n_z)
        self.pred_enc_gcn = GNNLayer(n_z, n_clusters)

        # 融合用GCN网络还原
        self.pred_dec_gcn = GNNLayer(n_clusters, n_z)
        self.gcn_n_dec_1 = GNNLayer(n_z, n_dec_1)
        self.gcn_n_dec_2 = GNNLayer(n_dec_1, n_dec_2)
        self.gcn_n_dec_3 = GNNLayer(n_dec_2, n_dec_3)
        self.gcn_n_dec_4 = GNNLayer(n_dec_3, n_input)
        # ---------------------------------------
        # GAT
        self.gat_n_enc_1 = GraphAttentionLayer(n_input, n_enc_1, alpha)
        self.gat_n_enc_2 = GraphAttentionLayer(n_enc_1, n_enc_2, alpha)
        self.gat_n_enc_3 = GraphAttentionLayer(n_enc_2, n_enc_3, alpha)
        self.gat_n_enc_4 = GraphAttentionLayer(n_enc_3, n_z, alpha)
        self.pred_enc_gat = GraphAttentionLayer(n_z, n_clusters, alpha)

        self.pred_dec_gat = GraphAttentionLayer(n_clusters, n_z, alpha)
        self.gat_n_dec_1 = GraphAttentionLayer(n_z, n_dec_1, alpha)
        self.gat_n_dec_2 = GraphAttentionLayer(n_dec_1, n_dec_2, alpha)
        self.gat_n_dec_3 = GraphAttentionLayer(n_dec_2, n_dec_3, alpha)
        self.gat_n_dec_4 = GraphAttentionLayer(n_dec_3, n_input, alpha)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_clusters))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):

        # AE Module
        x_hat_ae, h, h_cluster, tra1, tra2, tra3 = self.ae(x)

        sigma = 0.5

        # GCN Module
        # z = self.gcn_n_enc_1(x, adj)
        # z = self.gcn_n_enc_2((1 - sigma) * z + sigma * tra1, adj)
        # z = self.gcn_n_enc_3((1 - sigma) * z + sigma * tra2, adj)
        # z = self.gcn_n_enc_4((1 - sigma) * z + sigma * tra3, adj)
        # z = self.pred_enc_gcn((1 - sigma) * z + sigma * h, adj, active=False)

        # predict = F.softmax(z, dim=1)

        # 还原
        # z_hat = self.pred_dec_gcn(z, adj)
        # z_hat = self.gcn_n_dec_1(z_hat, adj)
        # z_hat = self.gcn_n_dec_2(z_hat, adj)
        # z_hat = self.gcn_n_dec_3(z_hat, adj)
        # z_hat = self.gcn_n_dec_4(z_hat, adj)

        # GAT
        z, _ = self.gat_n_enc_1(x, adj, M=None)
        z, _ = self.gat_n_enc_2((1 - sigma) * z + sigma * tra1, adj, M=None)
        z, _ = self.gat_n_enc_3((1 - sigma) * z + sigma * tra2, adj, M=None)
        z, _ = self.gat_n_enc_4((1 - sigma) * z + sigma * tra3, adj, M=None)
        z, _ = self.pred_enc_gat((1 - sigma) * z + sigma * h, adj, M=None)

        z_hat, _ = self.pred_dec_gat(z, adj, M=None)
        z_hat, _ = self.gat_n_dec_1(z_hat, adj, M=None)
        z_hat, _ = self.gat_n_dec_2(z_hat, adj, M=None)
        z_hat, _ = self.gat_n_dec_3(z_hat, adj, M=None)
        z_hat, _ = self.gat_n_dec_4(z_hat, adj, M=None)

        adj_hat_ae = torch.sigmoid(torch.mm(h, h.t()))
        adj_hat = torch.sigmoid(torch.mm(z, z.t()))

        # H→Q→P
        q_h = 1.0 / (1.0 + torch.sum(torch.pow(h_cluster.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_h = q_h.pow((self.v + 1.0) / 2.0)
        q_h = (q_h.t() / torch.sum(q_h, 1)).t()
        # Z→Q→P
        q_z = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_z = q_z.pow((self.v + 1.0) / 2.0)
        q_z = (q_z.t() / torch.sum(q_z, 1)).t()

        return x_hat_ae, adj_hat_ae, z_hat, adj_hat, z, q_h, q_z