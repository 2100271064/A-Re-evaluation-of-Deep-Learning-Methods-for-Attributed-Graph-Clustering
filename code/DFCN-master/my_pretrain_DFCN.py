# 训练DFCN网络/训练DCRN网络 epoch=100
import opt
from AE import *
from IGAE_mod import *
from DFCN import *
from load_data_mod import *
from utils_mod import *
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# GPU是否开启
if torch.cuda.is_available():
    print("Available GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

dataset_name = opt.args.name
root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset_name)
x = np.loadtxt(root_path + '/feature.txt', dtype=float)
vertex_arr = np.array(pd.read_table(root_path + "/node.txt", header=None)[0])
x = x[vertex_arr]

# TODO 原来是打开的
# # 其他数据集均n_input=100(参照DFCN)
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

# 预训练DFCN模型
def pretrain_DFCN(model, data, adj, label, n_outliers, lr, pre_model_gae_path, pre_model_igae_path, n_clusters,
          gamma_value, lambda_value, device):

    # 检查是否有用gpu
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("当前设备索引：", torch.cuda.current_device())
    else:
        print("no cude", device.type)

    print("\n", model)
    optimizer = Adam(model.parameters(), lr=lr)
    # 加载预训练模型
    model.ae.load_state_dict(torch.load(pre_model_gae_path, map_location='cpu'))
    model.gae.load_state_dict(torch.load(pre_model_igae_path, map_location='cpu'))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(data, adj)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(label, n_outliers, cluster_id, 'Initialization')

    for epoch in range(100):
        
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = model(data, adj)

        tmp_q = q.data
        p = target_distribution(tmp_q)

        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_igae = loss_w + gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + lambda_value * loss_kl
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        acc, nmi, ari, f1 = eva(label, n_outliers, kmeans.labels_, epoch)
        print("acc: {:.4f}, nmi: {:.4f}, ari: {:.4f}, f1: {:.4f}".format(acc, nmi, ari, f1))

        # 预训练的DFCN的存放路径
        pre_model_save_path = '/home/laixy/AHG/data/DFCN-master/model_pre/{}_DFCN.pkl'.format(dataset_name)
        torch.save(model.state_dict(), pre_model_save_path)

# citeseer、acm、dblp按照论文设置的学习率, 其他的默认1e-4
lr = 1e-4
if dataset_name == 'acm':
    lr = 5e-5

# 预训练模型的根目录路径
pre_model_root_path = '/home/laixy/AHG/data/DFCN-master/model_pre/{}_'.format(dataset_name)

y = np.loadtxt(root_path + '/label.txt', dtype=int)
[opt.args.n_clusters, n_outliers] = cal_n_cluster(opt.args.name)

model = DFCN(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
        n_input=opt.args.n_components,
        n_z=opt.args.n_z,
        n_clusters=opt.args.n_clusters,
        v=opt.args.freedom_degree,
        n_node=data.size()[0],
        device=device).to(device)
        
pre_model_gae_path = pre_model_root_path + 'gae.pkl'
pre_model_igae_path = pre_model_root_path + 'igae.pkl'
pretrain_DFCN(model, data, adj, y, n_outliers, lr, pre_model_gae_path, pre_model_igae_path, n_clusters=opt.args.n_clusters, \
                gamma_value=opt.args.gamma_value, lambda_value=opt.args.lambda_value, device=device)