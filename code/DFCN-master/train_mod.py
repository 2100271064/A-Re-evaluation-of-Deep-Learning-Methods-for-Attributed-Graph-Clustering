'''
Author: xinying Lai
Date: 2022-09-12 22:13:37
LastEditTime: 2022-09-13 10:32:25
Description: Do not edit
'''
import opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
# from utils_mod import adjust_learning_rate
from utils_mod import eva, target_distribution
import numpy as np


# use_adjust_lr = ['usps', 'hhar', 'reut', 'acm', 'dblp', 'cite']


def Train(epoch, model, data, adj, label, n_outliers, lr, pre_model_save_path, final_model_save_path, n_clusters,
          original_acc, gamma_value, lambda_value, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(pre_model_save_path, map_location='cpu'))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(data, adj)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(label, n_outliers, cluster_id, 'Initialization')

    acc_reuslt = []
    nmi_result = []
    ari_result = []
    f1_result = []
    for epoch in range(epoch):
        # if opt.args.name in use_adjust_lr:
        #     adjust_learning_rate(optimizer, epoch)
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
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        # if acc > original_acc:
        #     original_acc = acc
        # torch.save(model.state_dict(), final_model_save_path)

    print("Optimization Finished!")
    acc_arr = np.array(acc_reuslt)
    nmi_arr = np.array(nmi_result)
    ari_arr = np.array(ari_result)
    f1_arr = np.array(f1_result)
    
    return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr)
