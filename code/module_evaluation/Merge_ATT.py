import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import opt
from AE import *
from GCN import *
from ATT_AE2 import *

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

class Merge_ATT(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_dec_1, n_dec_2, n_dec_3, n_z,
                 n_input, n_clusters, alpha, v=1.0):
        super(Merge_ATT, self).__init__()

        # autoencoder for intra information
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

        # # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_3 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_5 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_7 = GNNLayer(n_enc_3, n_z)
        self.gnn_9 = GNNLayer(n_z, n_clusters)

        self.gnn_10 = GNNLayer(n_clusters, n_z)
        self.gnn_11 = GNNLayer(n_z, n_dec_1)
        self.gnn_12 = GNNLayer(n_dec_1, n_dec_2)
        self.gnn_13 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_14 = GNNLayer(n_dec_3, n_input)

        # GAT
        self.gat_1 = GraphAttentionLayer(n_input, n_enc_1, alpha)
        self.gat_3 = GraphAttentionLayer(n_enc_1, n_enc_2, alpha)
        self.gat_5 = GraphAttentionLayer(n_enc_2, n_enc_3, alpha)
        self.gat_7 = GraphAttentionLayer(n_enc_3, n_z, alpha)
        self.gat_9 = GraphAttentionLayer(n_z, n_clusters, alpha)

        self.gat_10 = GraphAttentionLayer(n_clusters, n_z, alpha)
        self.gat_11 = GraphAttentionLayer(n_z, n_dec_1, alpha)
        self.gat_12 = GraphAttentionLayer(n_dec_1, n_dec_2, alpha)
        self.gat_13 = GraphAttentionLayer(n_dec_2, n_dec_3, alpha)
        self.gat_14 = GraphAttentionLayer(n_dec_3, n_input, alpha)

        self.attn2 = SelfAttentionWide(n_enc_1)
        self.attn4 = SelfAttentionWide(n_enc_2)
        self.attn6 = SelfAttentionWide(n_enc_3)
        self.attn8 = SelfAttentionWide(n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_clusters))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_hat, h, h_cluster, tra1, tra2, tra3 = self.ae(x)

        # GCN Module
        # z = self.gnn_1(x, adj)
        # GAT
        z, _ = self.gat_1(x, adj, M=None)
        # 融合
        z = self.attn2((z + tra1))
        # 上面返回三维，第一维是1，去掉它，重新变成二维
        z = z.squeeze(0)

        # z = self.gnn_3(z, adj)
        z, _ = self.gat_3(z, adj, M=None)
        z = self.attn4((z + tra2))
        z = z.squeeze(0)

        # z = self.gnn_5(z, adj)
        z, _ = self.gat_5(z, adj, M=None)
        z = self.attn6((z + tra3))
        z = z.squeeze(0)

        # z = self.gnn_7(z, adj)
        z, _ = self.gat_7(z, adj, M=None)
        z = self.attn8((z + h))
        z = z.squeeze(0)

        # z = self.gnn_9(z, adj, active=False)
        z, _ = self.gat_9(z, adj, M=None)

        # predict = F.softmax(z, dim=1)

        # GCN
        # z_hat = self.gnn_10(z, adj)
        # z_hat = self.gnn_11(z_hat, adj)
        # z_hat = self.gnn_12(z_hat, adj)
        # z_hat = self.gnn_13(z_hat, adj)
        # z_hat = self.gnn_14(z_hat, adj)
        # GAT
        z_hat, _ = self.gat_10(z, adj, M=None)
        z_hat, _ = self.gat_11(z_hat, adj, M=None)
        z_hat, _ = self.gat_12(z_hat, adj, M=None)
        z_hat, _ = self.gat_13(z_hat, adj, M=None)
        z_hat, _ = self.gat_14(z_hat, adj, M=None)

        adj_hat_ae = torch.sigmoid(torch.mm(h, h.t()))
        A_pred = torch.sigmoid(torch.mm(z, z.t()))
        # print("forward:A_pred", A_pred)

        # H→Q→P
        q_h = 1.0 / (1.0 + torch.sum(torch.pow(h_cluster.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_h = q_h.pow((self.v + 1.0) / 2.0)
        q_h = (q_h.t() / torch.sum(q_h, 1)).t()
        # Z→Q→P
        q_z = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_z = q_z.pow((self.v + 1.0) / 2.0)
        q_z = (q_z.t() / torch.sum(q_z, 1)).t()

        return x_hat, adj_hat_ae, z_hat, A_pred, z, q_h, q_z