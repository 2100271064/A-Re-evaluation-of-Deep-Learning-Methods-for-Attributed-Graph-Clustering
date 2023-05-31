# 训练IGAE子网络 epoch=30
import opt
from IGAE_mod import *
from load_data_mod import *
from utils_mod import *
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
# GPU是否开启
if torch.cuda.is_available():
    print("Available GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

dataset_name = opt.args.name
root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset_name)
x = np.loadtxt(root_path + '/feature.txt', dtype=float)
vertex_arr = np.array(pd.read_table(root_path + "/node.txt", header=None)[0])
x = x[vertex_arr]

# TODO 原来是打开的
# 其他数据集均n_input=100(参照DFCN)
# if dataset_name == "dblp":
#     n_input = 50
# else:
#     n_input = 100
if (opt.args.name == "dblp"):
    opt.args.n_components = 256
pca = PCA(n_components=opt.args.n_components, svd_solver='full')
X_pca = pca.fit_transform(x)

dataset = LoadDataset(X_pca)

# 内含对vertex_arr的筛选
adj = load_graph(dataset_name).to(device)
data = torch.Tensor(dataset.x).to(device)

# 预训练IGAE模型
def pretrain_igae(model, data, adj, y, n_outliers, lr, n_clusters, device):

    # 检查是否有用gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("当前设备索引：", torch.cuda.current_device())
    else:
        print("no cude", device.type)

    print("\n", model)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(30):

        z_igae, z_hat, adj_hat = model(data, adj)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_igae = loss_w + opt.args.gamma_value * loss_a
        loss = loss_igae
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_igae.data.cpu().numpy())
        eva(y, n_outliers, kmeans.labels_, epoch)

        torch.save(model.state_dict(), '/home/laixy/AHG/data/DFCN-master/model_pre/{0}_igae.pkl'.format(dataset_name))

# citeseer、acm、dblp按照论文设置的学习率, 其他的默认1e-4
lr = 1e-4
if dataset_name == 'acm':
    lr = 5e-5


y = np.loadtxt(root_path + '/label.txt', dtype=int)

model = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1,
        gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_enc_3=opt.args.gae_n_enc_3,
        gae_n_dec_1=opt.args.gae_n_dec_1,
        gae_n_dec_2=opt.args.gae_n_dec_2,
        gae_n_dec_3=opt.args.gae_n_dec_3,
        n_input=opt.args.n_components
        ).to(device)

[n_clusters, n_outliers] = cal_n_cluster(dataset_name)
pretrain_igae(model, data, adj, y, n_outliers, lr=lr, n_clusters=n_clusters, device=device)
