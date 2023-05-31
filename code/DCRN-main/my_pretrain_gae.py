import opt
from DCRN import AE_encoder, AE_decoder
from utils_mod import *
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Auto Encoder from DFCN
class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)
    
    # 自己加的
    def forward(self, x):
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae

# GPU是否开启
if torch.cuda.is_available():
    print("Available GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cuda")

dataset_name = opt.args.name
root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset_name)
x = np.loadtxt(root_path + '/feature.txt', dtype=float)
vertex_arr = np.array(pd.read_table(root_path + "/node.txt", header=None)[0])
x = x[vertex_arr]

# TODO 原来是打开的
# if opt.args.name == "dblp":
#     n_input = 50
# else:
#     n_input = 100
if (opt.args.name == "dblp"):
    opt.args.n_components = 256
pca = PCA(n_components=opt.args.n_components, svd_solver='full')
X_pca = pca.fit_transform(x)

data = torch.Tensor(X_pca).to(device)

# citeseer、acm、dblp、pubmed按照论文设置的学习率, 其他的默认1e-5
lr = 1e-5
if opt.args.name == 'acm':
    lr = 5e-5
elif opt.args.name == 'dblp':
    lr = 1e-4


y = np.loadtxt(root_path + '/label.txt', dtype=int)

def pretrain_gae(model, data, y):

    # 检查是否有用gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("当前设备索引：", torch.cuda.current_device())
    else:
        print("no cude", device.type)

    print("\n", model)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)

        x_bar, z = model(data)
        loss = F.mse_loss(x_bar, data)
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        [n_clusters, n_outliers] = cal_n_cluster(dataset_name)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z.data.cpu().numpy())
        eva(y, n_outliers, kmeans.labels_, epoch)

        torch.save(model.state_dict(), '/home/laixy/AHG/data/DCRN-main/model_pre/{0}_gae.pkl'.format(dataset_name))

model = AE(
    ae_n_enc_1=opt.args.ae_n_enc_1,
    ae_n_enc_2=opt.args.ae_n_enc_2,
    ae_n_enc_3=opt.args.ae_n_enc_3,
    ae_n_dec_1=opt.args.ae_n_dec_1,
    ae_n_dec_2=opt.args.ae_n_dec_2,
    ae_n_dec_3=opt.args.ae_n_dec_3,
    # 维度
    n_input=opt.args.n_components,
    n_z=opt.args.n_z).to(device)

pretrain_gae(model, data, y)