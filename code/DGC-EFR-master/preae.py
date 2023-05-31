import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='acm')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_clusters', default=3, type=int)
# parser.add_argument('--n_z', default=128, type=int)
parser.add_argument('--pretrain_path', type=str, default='pkl')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--n_components', type=int, default=500)
# parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
# parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
args = parser.parse_args()
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
from utils_mod import *
from sklearn.decomposition import PCA

# the basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # encoder configuration
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder configuration
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
        # degree
        self.v = v

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

# pre-train the autoencoder model
def pretrain_ae(model, dataset, args, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    for epoch in range(args.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            eva(y, args.n_outliers, kmeans.labels_, epoch)

        root_path = r"/home/laixy/AHG/data/DGC-EFR-master"
        torch.save(model.state_dict(), root_path + r'/model_pre/{}_ae.pkl'.format(args.name))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='train',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--name', type=str, default='acm')
    # parser.add_argument('--k', type=int, default=3)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--n_clusters', default=3, type=int)
    # # parser.add_argument('--n_z', default=128, type=int)
    # parser.add_argument('--pretrain_path', type=str, default='pkl')
    # # parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    # # parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    # args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    root_path = r"/home/laixy/AHG"
    args.pretrain_path = root_path + r'/data/DGC-EFR-master/model_pre/{}_ae.pkl'.format(args.name)

    x_path = root_path + r'/dataset/{}/feature.txt'.format(args.name)
    y_path = root_path + r'/dataset/{}/label.txt'.format(args.name)
    node_path = root_path + r'/dataset/{}/node.txt'.format(args.name)
    x = np.loadtxt(x_path, dtype=float)
    y = np.loadtxt(y_path, dtype=int)
    node = np.loadtxt(node_path, dtype=int)
    x = x[node]

    # TODO 特征工程
    if (args.name == "dblp"):
        args.n_components = 256
    pca = PCA(n_components=args.n_components, svd_solver='full')
    x = pca.fit_transform(x)
    print("PCA后,feature.shape=", x.shape)

    args.k = None
    args.n_input = x.shape[1]
    [args.n_clusters, args.n_outliers] = cal_n_cluster(args.name)

    model = AE(
            n_enc_1=1024,
            n_enc_2=256,
            n_enc_3=16,
            n_dec_1=16,
            n_dec_2=256,
            n_dec_3=1024,
            n_input=args.n_input,
            n_z=10,).cuda()

    dataset = LoadDataset(x)
    pretrain_ae(model, dataset, args, y)
