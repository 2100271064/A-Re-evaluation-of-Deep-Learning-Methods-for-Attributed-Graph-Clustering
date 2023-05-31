#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from preprocessing import sparse_to_tuple
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from munkres import Munkres

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def q_mat(X, centers, alpha=1.0):
    X = X.detach().numpy()
    centers = centers.detach().numpy()
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q ** ((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q
  
def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0)
    # confidence1 = q.max(1) # citeseer
    confidence1 = np.zeros((q.shape[0],)) # cora. pubmed
    confidence2 = np.zeros((q.shape[0],))
    a = np.argsort(q, axis=1)
    for i in range(q.shape[0]):
        confidence1[i] = q[i,a[i,-1]]
        confidence2[i] = q[i,a[i,-2]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

class clustering_metrics():
    def __init__(self, true_label, predict_label, n_outliers):
        print("\n删除n_outliers前predict_label：", predict_label.shape)
        # 删除无效点
        if n_outliers > 0:
            true_label = true_label[:-n_outliers]
            predict_label = predict_label[:-n_outliers]
        self.true_label = true_label
        self.pred_label = predict_label
        print("删除n_outliers后predict_label：", self.pred_label.shape)

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return -1, -1, -1, -1, -1, -1, -1

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()
        if acc == -1:
            return -1, -1, -1, -1, -1, -1, -1
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        # fh = open('recoder.txt', 'a')
        #
        # fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        # fh.write('\r\n')
        # fh.flush()
        # fh.close()

        return acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class ReGMM_VGAE(nn.Module):
    def __init__(self, **kwargs):
        super(ReGMM_VGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']

        # VGAE training parameters
        self.base_gcn = GraphConvSparse( self.num_features, self.num_neurons)
        self.gcn_mean = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)

        # GMM training parameters    
        self.pi = nn.Parameter(torch.ones(self.nClusters)/self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)

        # 自己加的
        self.dataset = kwargs['dataset']

    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, epochs, lr, save_path, dataset):
        opti = Adam(self.parameters(), lr=lr)
        epoch_bar = tqdm(range(epochs))
        gmm = GaussianMixture(n_components = self.nClusters , covariance_type = 'diag')
        # #------------------------------------------------------------#
        # dense_tensor = features.to_dense()  # 转为密集矩阵
        # dense_tensor = dense_tensor / (dense_tensor.max() - dense_tensor.min()) * 10  # 将大于 3.0 的元素设为 10.0
        # values = dense_tensor[dense_tensor >= 0]  # 获取新的 values
        # indices = torch.nonzero(dense_tensor >= 0).t()  # 获取新的 indices
        # features = torch.sparse_coo_tensor(indices, values, dense_tensor.size())
        #
        # dense_tensor1 = adj.to_dense()  # 转为密集矩阵
        # dense_tensor1 = dense_tensor1 / (dense_tensor1.max() - dense_tensor1.min()) * 10  # 将大于 3.0 的元素设为 10.0
        # values = dense_tensor1[dense_tensor1 >= 0]  # 获取新的 values
        # indices = torch.nonzero(dense_tensor1 >= 0).t()  # 获取新的 indices
        # adj = torch.sparse_coo_tensor(indices, values, dense_tensor1.size())
        # # ------------------------------------------------------------#
        for _ in epoch_bar:
            opti.zero_grad()
            _,_, z = self.encode(features, adj)
            x_ = self.decode(z, self.dataset)
            # TODO wiki x_中出现nan值
            loss = norm * F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            loss.backward()
            opti.step()
        # gmm.fit_predict(z.detach().numpy()) # citeseer
        gmm.fit(z.detach().numpy()) # cora, pubmed
        self.pi.data = torch.from_numpy(gmm.weights_)
        self.mu_c.data = torch.from_numpy(gmm.means_)
        self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_))
        self.logstd = self.mean
        # 自己加的
        torch.save(self.state_dict(), save_path + '/pretrain/test/{}.pk'.format(dataset))

    def ELBO_Loss(self, features, adj, x_, adj_label, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        det = 1e-2
        Loss = 1e-2 * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight = weight_tensor)
        Loss = Loss * features.size(0)
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(emb,mu_c,log_sigma2_c))+det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))
        KL1 = 0.5 * torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        Loss1 = KL1 
        KL2 = torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        Loss1 -= KL2
        return Loss, Loss1, Loss+Loss1

    def generate_centers(self, emb_unconf):
        y_pred = self.predict(emb_unconf)
        nn = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(emb_unconf.detach().numpy())
        _, indices = nn.kneighbors(self.mu_c.detach().numpy())
        return indices[y_pred]

    def update_graph(self, adj, labels, emb, unconf_indices, conf_indices):
        k = 0
        y_pred = self.predict(emb)
        emb_unconf = emb[unconf_indices]
        adj = adj.tolil()
        idx = unconf_indices[self.generate_centers(emb_unconf)]    
        for i, k in enumerate(unconf_indices):
            adj_k = adj[k].tocsr().indices
            if not(np.isin(idx[i], adj_k)) and (y_pred[k] == y_pred[idx[i]]) :
                adj[k, idx[i]] = 1
            for j in adj_k:
                if np.isin(j, unconf_indices) and (np.isin(idx[i], adj_k)) and (y_pred[k] != y_pred[j]):
                    adj[k, j] = 0
        adj = adj.tocsr()
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                    torch.FloatTensor(adj_label[1]),
                                    torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() 
        weight_tensor[weight_mask] = pos_weight_orig
        return adj, adj_label, weight_tensor

    def train(self, adj_norm, adj, features, y, n_outliers, norm, epochs, lr, beta1, beta2, save_path, dataset,
              weight_decay, t1, stable, b1, b2, t2, min_epoch_t):
        self.load_state_dict(torch.load(save_path + '/pretrain/test/{}.pk'.format(dataset)))
        # weight_decay = 0.089 # citeseer
        # weight_decay = 0.0001 # cora
        # weight_decay = 0.001 # pubmed
        opti = Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        import os, csv
        epoch_bar = tqdm(range(epochs))
        epoch_stable = 0
        previous_unconflicted = []
        previous_conflicted = []

        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm) 
            x_ = self.decode(emb, self.dataset)
            if epoch % t1 == 0: # 1 for citeseer, 10 for cora, 5 for pubmed
                unconflicted_ind, conflicted_ind = generate_unconflicted_data_index(emb, self.mu_c, beta1, beta2)
                if epoch == 0:
                    adj, adj_label, weight_tensor = self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            if len(previous_unconflicted) < len(unconflicted_ind) :
                z_mu = z_mu[unconflicted_ind]
                z_sigma2_log = z_sigma2_log[unconflicted_ind]
                emb_unconf = emb[unconflicted_ind]
                emb_conf = emb[conflicted_ind] #虽然和pubmed不一样，但是用不上
                previous_conflicted = conflicted_ind
                previous_unconflicted = unconflicted_ind
            else :
                epoch_stable += 1
                z_mu = z_mu[previous_unconflicted]
                z_sigma2_log = z_sigma2_log[previous_unconflicted]
                emb_unconf = emb[previous_unconflicted]
                emb_conf = emb[previous_conflicted] #虽然和pubmed不一样，但是用不上
            if epoch_stable >= stable: # 15 for citeseer,cora, 20 for pubmed
                epoch_stable = 0
                # b1 = 0.96 # 0.95 for cora, pubmed
                # b2 = 0.98 # 0.85 for cora, pubmed
                beta1 = beta1 * b1
                beta2 = beta2 * b2
            if epoch % t2 == 0 and epoch <= min_epoch_t : # t2 = 50 && min_epoch_t = 200 for citeseer, t2 = 20 && min_epoch_t = 120 for cora, t2 = 50 && min_epoch_t = 100 for pubmed
                adj, adj_label, weight_tensor = self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            loss, loss1, elbo_loss = self.ELBO_Loss(features, adj_norm, x_, adj_label.to_dense().view(-1), weight_tensor, norm, z_mu , z_sigma2_log, emb_unconf)
            epoch_bar.write('Loss={:.4f}'.format(elbo_loss.detach().numpy()))
            y_pred = self.predict(emb)                            
            cm = clustering_metrics(y, y_pred, n_outliers)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            if acc != -1:
                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(adjscore)
                f1_list.append(f1_macro)

            elbo_loss.backward()
            opti.step()
            lr_s.step()

        return np.mean(acc_list), np.mean(nmi_list), np.mean(ari_list), np.mean(f1_list), len(acc_list)

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def predict(self, z):
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c  
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita = yita_c.detach().numpy()
        return np.argmax(yita, axis=1)

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z
            
    @staticmethod
    def decode(z, dataset=None):
        # def scale_z(x):
        #     # 找到张量中的最大值和最小值
        #     min_value = torch.min(x)
        #     max_value = torch.max(x)
        #     # 计算最大值和最小值之间的差值
        #     diff = max_value - min_value
        #     # 将张量中的每个元素减去最小值，并将结果除以差值的一半
        #     x_scaled = (x - min_value) / (diff / 2)
        #     # 将张量中的每个元素乘以2，再减去1
        #     x_scaled = x_scaled * 2 - 1
        #     return x_scaled
        # if dataset == 'wiki':
        #     z = nn.functional.normalize(z, p=2, dim=1)
        # scale_z = scale_z(z)
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        # A_pred = torch.matmul(z, z.t())

        return A_pred