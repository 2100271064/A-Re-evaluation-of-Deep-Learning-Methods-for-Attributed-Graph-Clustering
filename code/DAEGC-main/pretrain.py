import argparse

parser = argparse.ArgumentParser(
    description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--name", type=str, default="uk")
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--n_clusters", default=6, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--embedding_size", default=16, type=int)
parser.add_argument("--weight_decay", type=int, default=5e-3)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--n_components', type=int, default=500)
args = parser.parse_args()

# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from utils_mod import *
from model import GAT
from evaluation import eva
from sklearn.decomposition import PCA


def pretrain(x, y, adj_label, adj):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    if args.name != "us" and args.name != "uk":
        adj = torch.from_numpy(adj).to(dtype=torch.float).to(device)
        adj_label = torch.Tensor(adj_label)
        M = get_M(adj).to(device)
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        adj_label = sparse_mx_to_torch_sparse_tensor(adj_label)
        M = get_M(adj.to_dense()).to(device)
    x = torch.Tensor(x).to(device)

    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1).cpu(), adj_label.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, args.n_outliers, kmeans.labels_, epoch)

        torch.save(model.state_dict(), '/home/laixy/AHG/data/DAEGC-main/model_pre/{0}.pkl'.format(args.name))



if __name__ == "__main__":

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    [args.n_clusters, args.n_outliers] = cal_n_cluster(args.name)
    args.lr = 0.0001

    if args.name == "pubmed":
        args.lr = 0.001

    features, labels, adj_label, adj = load_data(args.name)

    # TODO 特征工程
    # if (args.name == "dblp"):
    #     args.n_components = 256
    # pca = PCA(n_components=args.n_components, svd_solver='full')
    # features = pca.fit_transform(features)

    args.input_dim = features.shape[1]

    print(args)
    pretrain(features, labels, adj_label, adj)
