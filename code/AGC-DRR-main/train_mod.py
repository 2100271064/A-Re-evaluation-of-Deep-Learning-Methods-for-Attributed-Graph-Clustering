import torch
from opt import args
from utils_mod import eva#,target_distribution
from torch.optim import Adam
import torch.nn.functional as F
from load_data import *

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
acc_reuslt = []
# acc_reuslt.append(0)
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Train_gae(model,view_learner, data, adj, label, n_outliers, edge_index, device):
    acc_reuslt = []
    # acc_reuslt.append(0)
    nmi_result = []
    ari_result = []
    f1_result = []

    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):

        # loss1

        view_learner.train()
        view_learner.zero_grad()
        model.eval()
        # 对抗网络1
        z_igae, c= model(data, adj)

        n = z_igae.shape[0]
        # 边权重 shape=(M, 1)
        edge_logits = view_learner(data,adj,edge_index)

        # 将边权重 0-1化
        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

        # 权重矩阵
        if opt.args.name != "us" and opt.args.name != "uk":
            aug_adj= new_graph(torch.tensor(edge_index).to('cuda'),batch_aug_edge_weight,n,'cuda')
            aug_adj = aug_adj.to_dense()
            # 增强邻接矩阵
            aug_adj = aug_adj * adj
            aug_adj = aug_adj.cpu().detach().numpy()+np.eye(n)
            aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to('cuda')
        else:
            aug_adj = new_graph(torch.tensor(edge_index).to('cuda'), batch_aug_edge_weight, n, 'cuda')
            # aug_adj = torch.sparse.mul(aug_adj, adj)
            # 逐元素相乘
            aug_adj_coo = aug_adj.coalesce()
            adj_coo = adj.coalesce()
            values_ = torch.sparse.FloatTensor(aug_adj_coo.size(), device="cpu").to(device)
            values_ = values_.coalesce()
            for i, j, v1 in zip(aug_adj_coo.indices()[0], aug_adj_coo.indices()[1], aug_adj_coo.values()):
                v2 = adj_coo[i, j]
                if v2 != 0:
                    values_[i, j] = v1 * v2
            aug_adj = values_
            # aug_adj_coo = aug_adj.coalesce()
            # adj_coo = adj.coalesce()
            # # empty = torch.sparse.FloatTensor(aug_adj_coo.size(), device="cpu").to(device)
            # # 获取行、列下标和元素值
            # row, col = aug_adj_coo.indices()[0], aug_adj_coo.indices()[1]
            # new_aug_adj = torch.Tensor.clone(aug_adj)
            # values = new_aug_adj.coalesce().values()
            # for i, j, v1 in zip(aug_adj_coo.indices()[0], aug_adj_coo.indices()[1], aug_adj_coo.values()):
            #     v2 = adj_coo[i, j]
            #     if v2 != 0:
            #         # 找到需要修改的元素的下标
            #         idx = (row == i) & (col == j)
            #         # 修改对应下标的元素值
            #         values[idx] = v1 * v2
            # aug_adj = torch.sparse_coo_tensor(new_aug_adj.indices(), values, new_aug_adj.size()).to_sparse()
            # 创建一个形状与稀疏张量相同的单位对角矩阵
            eye = torch.sparse.eye(aug_adj.shape[0], dtype=aug_adj.dtype, device=aug_adj.device)
            # 稀疏张量加上单位对角矩阵
            aug_adj = aug_adj + eye
            aug_adj = normalize_sparse(aug_adj)

        # 对抗网络2
        aug_z_igae,aug_c= model(data, aug_adj)


        edge_drop_out_prob = 1 - batch_aug_edge_weight
        reg = edge_drop_out_prob.mean()

        view_loss = (args.reg_lambda * reg)+model.calc_loss(c.T,aug_c.T)+model.calc_loss(c, aug_c)

        (-view_loss).backward()
        view_optimizer.step()

        view_learner.eval()

        # loss2

        model.train()
        model.zero_grad()
        z_igae, c = model(data, adj)

        n = z_igae.shape[0]
        #with torch.no_grad():
        edge_logits = view_learner(data, adj, edge_index)


        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

        if opt.args.name != "us" and opt.args.name != "uk":
            aug_adj = new_graph(torch.tensor(edge_index).to('cuda'), batch_aug_edge_weight, n, 'cuda')
            aug_adj = aug_adj.to_dense()
            # 增强邻接矩阵
            aug_adj = aug_adj * adj
            aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
            aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to('cuda')
        else:
            aug_adj = new_graph(torch.tensor(edge_index).to('cuda'), batch_aug_edge_weight, n, 'cuda')
            # aug_adj = torch.sparse.mul(aug_adj, adj)
            # 逐元素相乘
            adu_adj_coo = aug_adj.coalesce()
            adj_coo = adj.coalesce()
            for i, j, v1 in zip(adu_adj_coo.indices()[0], adu_adj_coo.indices()[1], adu_adj_coo.values()):
                v2 = adj_coo[i, j]
                if v2 != 0:
                    aug_adj[i, j] = v1 * v2
            # 创建一个形状与稀疏张量相同的单位对角矩阵
            eye = torch.sparse.eye(aug_adj.shape[0], dtype=aug_adj.dtype, device=aug_adj.device)
            # 稀疏张量加上单位对角矩阵
            aug_adj = aug_adj + eye
            aug_adj = normalize_sparse(aug_adj)

        aug_z_igae, aug_c = model(data, aug_adj)

        z_mat =torch.matmul(z_igae, aug_z_igae.T)


        model_loss = model.calc_loss(c.T, aug_c.T) + F.mse_loss(z_mat, torch.eye(n).to('cuda'))+ model.calc_loss(c, aug_c)
        model_loss.backward()
        optimizer.step()
        model.eval()

        print('{} loss: {}'.format(epoch, model_loss))

        # 计算指标

        z = (c + aug_c)/2
     #   kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
        i = z.argmax(dim=-1)
        acc, nmi, ari, f1 = eva(label, n_outliers, i.data.cpu().numpy(), epoch)
        if acc != -1:
    #    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
            acc_reuslt.append(acc)
            nmi_result.append(nmi)
            ari_result.append(ari)
            f1_result.append(f1)
    
    return np.mean(acc_reuslt), np.mean(nmi_result), np.mean(ari_result), np.mean(f1_result), len(acc_reuslt)

