from run_pubmed import args
from utils_mod import *
from model import GATE
from evaluation import eva
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA


# GPU是否开启
if torch.cuda.is_available():
    print("Available GPU")
    # device = torch.device("cuda")
    args.cuda = True
else:
    print("Using CPU")
    # device = torch.device("cuda")
    args.cuda = False

class LoadDataset:
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

dataset_name = args.name

setup(args)

adj, adj_normalized, features, label = load_data(args.name)

# TODO 特征工程
if (args.name == "dblp"):
    args.n_components = 256
pca = PCA(n_components=args.n_components, svd_solver='full')
features = pca.fit_transform(features)
print("PCA后,feature.shape=", features.shape)

args.input_dim = features.shape[1]

adj_normalized = adj_normalized.to_dense().to(args.device)
M = get_M(adj_normalized).to(args.device)

adj_label = torch.Tensor(adj).to(args.device)
# if args.n_outliers > 0:
#     adj_label = adj_label[:-args.n_outliers, :-args.n_outliers]

data = torch.Tensor(features).to(args.device)
y = label
adj = adj_normalized


lr = args.lr


def pretrain(model, data, adj, y, M):

    # 检查是否有用gpu
    if args.device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("当前设备索引：", torch.cuda.current_device())
    else:
        print("no cude", args.device.type)

    print("\n", model)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)

        # split batch
        index = list(range(len(data)))
        split_list = data_split(index, 2000)

        print("共有{}个batch".format(len(split_list)))

        batch_count = 0

        model.train()

        model = model.to(args.device)
        model.device = "cuda"
        data = data.to(args.device)
        adj = adj.to(args.device)
        M = M.to(args.device)

        for batch in split_list:

            batch_count = batch_count + 1
            data_batch = data[batch]
            adj_batch = adj[batch, :][:, batch]
            M_batch = M[batch, :][:, batch]

            A_pred, z = model(data_batch, adj_batch, M_batch)
            # A_pred = A_pred[:-args.n_outliers, :-args.n_outliers]
            loss = F.binary_cross_entropy(A_pred.contiguous().view(-1), adj_label[batch, :][:, batch].contiguous().view(-1))
            print("当前epoch={}, 当前batch={}, loss={}".format(epoch, batch_count, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_cpu = model.cpu()
        model_cpu.device = "cpu"
        data = data.cpu()
        adj = adj.cpu()
        M = M.cpu()
        with torch.no_grad():
            A_pred, z = model_cpu(data, adj, M)
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
        eva(y, args.n_outliers, kmeans.labels_, epoch)

        torch.save(model.state_dict(), '/home/laixy/AHG/data/CDBNE-master/model_pre/{0}.pkl'.format(dataset_name))

model = GATE(
    attribute_number=args.input_dim,
    hidden_size=args.middle_size,
    embedding_size=args.representation_size,
    alpha=args.alpha
    ).to(args.device)


pretrain(model, data, adj, y, M)