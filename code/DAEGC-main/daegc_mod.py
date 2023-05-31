import argparse

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='acm')
# parser.add_argument('--epoch', type=int, default=30)
# TODO epoch 原来是100
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_clusters', default=6, type=int)
parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--embedding_size', default=16, type=int)
parser.add_argument('--weight_decay', type=int, default=5e-3)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_components', type=int, default=500)
args = parser.parse_args()

# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam


from utils_mod import *
from model import GAT
from evaluation import eva
from sklearn.decomposition import PCA


class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get Citeseer model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        # TODO
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def trainer(x, y, adj_label, adj):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
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
    data = torch.Tensor(x).to(device)

    with torch.no_grad():
        _, z = model.gat(data, adj, M)

    # get kmeans and Citeseer cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, args.n_outliers, y_pred, 'Citeseer')

    acc_result = []
    nmi_result = []
    ari_result = []
    f1_result = []

    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)
            
            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(y, args.n_outliers, q, epoch)
            if acc != -1:
                acc_result.append(acc)
                nmi_result.append(nmi)
                ari_result.append(ari)
                f1_result.append(f1)

        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1).cpu(), adj_label.contiguous().view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(acc_result), np.mean(nmi_result), np.mean(ari_result), np.mean(f1_result), len(acc_result)

if __name__ == "__main__":

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    [args.n_clusters, args.n_outliers] = cal_n_cluster(args.name)
    # TODO 原来是打开的
    # args.lr = 0.0001
    #
    # if args.name == "pubmed":
    #     args.lr = 0.001

    # TODO 预训练
    args.pretrain_path = '/home/laixy/AHG/data/DAEGC-main/model_pre/{0}.pkl'.format(args.name)

    features, labels, adj_label, adj = load_data(args.name)

    # TODO 特征工程
    # if (args.name == "dblp"):
    #     args.n_components = 256
    # pca = PCA(n_components=args.n_components, svd_solver='full')
    # features = pca.fit_transform(features)
    # print("PCA后,feature.shape=", features.shape)

    args.input_dim = features.shape[1]


    print(args)

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    valid_epoch_num_list = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = trainer(features, labels, adj_label, adj)

        final_acc.append(mean_acc)
        final_nmi.append(mean_nmi)
        final_ari.append(mean_ari)
        final_f1.append(mean_f1)
        valid_epoch_num_list.append(valid_epoch_num)

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)
    print("{} epoch × 10, 有效的epoch数：".format(args.max_epoch), valid_epoch_num_list)

    value = np.mean(acc_arr)
    var = np.var(acc_arr)
    std = np.std(acc_arr)
    print('final_acc: {}, fianl_var_acc: {}, final_std_acc:{}'.format(value, var, std))
    print('final_acc: {:.4f}, fianl_var_acc: {:.2f}, final_std_acc:{:.2f}%'.format(value, var, std * 100))

    value = np.mean(nmi_arr)
    var = np.var(nmi_arr)
    std = np.std(nmi_arr)
    print('final_nmi: {}, final_var_nmi: {}, final_std_nmi:{}'.format(value, var, std))
    print('final_nmi: {:.4f}, final_var_nmi: {:.2f}, final_std_nmi:{:.2f}%'.format(value, var, std * 100))

    value = np.mean(ari_arr)
    var = np.var(ari_arr)
    std = np.std(ari_arr)
    print('final_ari: {}, final_var_ari: {}, final_std_ari:{}'.format(value, var, std))
    print('final_ari: {:.4f}, final_var_ari: {:.2f}, final_std_ari:{:.2f}%'.format(value, var, std * 100))

    value = np.mean(f1_arr)
    var = np.var(f1_arr)
    std = np.std(f1_arr)
    print('final_f1: {}, final_var_f1: {}, final_std_f1:{}'.format(value, var, std))
    print('final_f1: {:.4f}, final_var_f1: {:.2f}, final_std_f1:{:.2f}%'.format(value, var, std * 100))