import numpy as np
import h5py
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import pandas as pd
from sdcn_mod import args

# 原始预训练的evaluation.py
import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics

from utils_mod import *
from sklearn.decomposition import PCA

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def eva(y_true, n_outliers, y_pred, epoch=0):

    # 剥离outliers
    if n_outliers > 0:
        y_true = y_true[:-n_outliers]
        y_pred = y_pred[:-n_outliers]

    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
            ', f1 {:.4f}'.format(f1))

#torch.cuda.set_device(3)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


# def adjust_learning_rate(optimizer, epoch):
#     lr = 0.001 * (0.1 ** (epoch // 20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


dataset_name = args.name
root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset_name)

vertex_arr = np.array(pd.read_table(root_path + "/node.txt", header=None)[0])
if dataset_name != 'uk' and dataset_name != 'us':
    x = np.loadtxt(root_path + '/feature.txt', dtype=float)
else:
    x = np.load(root_path + '/feat192.npy')
x = x[vertex_arr]

# TODO 特征工程
# if (args.name == "dblp"):
#     args.n_components = 256
# pca = PCA(n_components=args.n_components, svd_solver='full')
# x = pca.fit_transform(x)

dataset = LoadDataset(x)
y = np.loadtxt(root_path + '/label.txt', dtype=int)


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x = torch.Tensor(dataset.x).cuda().float()
                x_bar, z = model(x)
                loss = F.mse_loss(x_bar, x)
                print('{} loss: {}'.format(epoch, loss))
                [n_clusters, n_outliers] = cal_n_cluster(dataset_name)
                kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z.data.cpu().numpy())
                eva(y, n_outliers, kmeans.labels_, epoch)

            torch.save(model.state_dict(), '/home/laixy/AHG/data/SDCN-master/model_pre/{0}.pkl'.format(dataset_name))

model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        # 维度
        n_input=x.shape[1],
        n_z=10,).cuda()

pretrain_ae(model, dataset, y)
