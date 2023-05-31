from torch import nn
from torch.nn import Linear


class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_z, n_input, n_clusters):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.n_z = Linear(ae_n_enc_3, n_z)
        self.enc_cluster = Linear(n_z, n_clusters)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        tra1 = self.act(self.enc_1(x))
        tra2 = self.act(self.enc_2(tra1))
        tra3 = self.act(self.enc_3(tra2))
        z_ae = self.n_z(tra3)
        z_ae_cluster = self.enc_cluster(z_ae)
        return z_ae, z_ae_cluster, tra1, tra2, tra3


class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_z, n_input):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.act(self.x_bar_layer(z))
        return x_hat


class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_z, n_input, n_clusters):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_z=n_z,
            n_input=n_input,
            n_clusters=n_clusters)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_z=n_z,
            n_input=n_input)

    def forward(self, x):
        z_ae, z_ae_cluster, tra1, tra2, tra3 = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae, z_ae_cluster, tra1, tra2, tra3