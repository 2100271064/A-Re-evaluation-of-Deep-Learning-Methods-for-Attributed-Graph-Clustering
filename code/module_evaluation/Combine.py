import torch
from torch import nn
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from AE import *
from GCN2 import *
from ATT_AE2 import *

class Combine(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_dec_1, n_dec_2, n_dec_3, n_z,
                 n_input, n_clusters, v=1.0, n_node=None, device=None):
        super(Combine, self).__init__()

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

        self.gcn = GCN2(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_z=n_z,
            n_input=n_input,
            n_clusters=n_clusters,
            v=v)

        self.gat = ATT_AE2(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_z=n_z,
            alpha=opt.args.alpha,
            n_input=n_input,
            n_clusters=n_clusters,
            v=v
        )


        # self.enc_cluster = Linear(gae_n_enc_3, n_clusters)
        if n_node is None:
            self.a = Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True).to(device)
        else:
            self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(device)
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x, adj):
        z_ae, _, _, _, _ = self.ae.encoder(x)
        # gcn
        # z_gcn, z_gcn_adj = self.gcn.encoder(x, adj)
        # gat
        z_gat_hat, z_gat_adj, z_gat, _, _, _, _ = self.gat(x, adj)
        z = self.a * z_ae + self.b * z_gat

        h_hat = self.ae.decoder(z_ae)
        # gcn
        # z_gcn_hat, _ = self.gcn.decoder(z_gat, adj)
        adj_hat_ae = torch.sigmoid(torch.mm(z_ae, z_ae.t()))
        # gcn
        # adj_hat_gcn = z_gcn_adj
        # gat如何解码最终z(拆开？)
        # z_hat, _ = self.gcn.decoder(z, adj)
        adj_hat = torch.sigmoid(torch.mm(z, z.t()))

        # H→Q→P
        q_h = 1.0 / (1.0 + torch.sum(torch.pow((z_ae).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_h = q_h.pow((self.v + 1.0) / 2.0)
        q_h = (q_h.t() / torch.sum(q_h, 1)).t()
        # Z→Q→P
        q_z = 1.0 / (1.0 + torch.sum(torch.pow((z).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_z = q_z.pow((self.v + 1.0) / 2.0)
        q_z = (q_z.t() / torch.sum(q_z, 1)).t()

        # return h_hat, adj_hat_ae, z_gcn_hat, adj_hat_gcn, z_hat, adj_hat, z_ae, z_gcn, z, q_h, q_z
        return h_hat, adj_hat_ae, z_gat_hat, z_gat_adj, None, adj_hat, z_ae, z_gat, z, q_h, q_z