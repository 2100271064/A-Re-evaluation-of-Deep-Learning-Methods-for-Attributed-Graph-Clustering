from __future__ import division
from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# For replicating the experiments
SEED = 42
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=35, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
# TODO lr 原本是0.001
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.0013, help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.7, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.8, help='Lower Threshold end.')
# TODO upd 原本是10
parser.add_argument('--upd', type=int, default=1, help='Update epoch.')
# TODO batch_size 原本是10000
parser.add_argument('--bs', type=int, default=1000, help='Batchsize.')
parser.add_argument('--dataset', type=str, default='dblp', help='type of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
# parser.add_argument('--version', type=str, default="max_cluster_num_5",
#                     help='max_cluster_num_version')
parser.add_argument('--gpu', type=str, default="3")
parser.add_argument('--n_components', type=int, default=500)
args = parser.parse_args()

import time
import random
import numpy as np
import scipy.sparse as sp
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# os一定要在torch前面
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch

np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from model import LinTrans#, LogReg
from optimizer import loss_function
from utils_mod import *
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda is True:
    print('Using GPU')
    torch.cuda.manual_seed(SEED)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def clustering(Cluster, feature, true_labels, n_outliers):

    f_adj = np.matmul(feature, np.transpose(feature))
    # try..
    predict_labels = Cluster.fit_predict(f_adj)

    # 剥离outliers
    if n_outliers > 0:
        true_labels = true_labels[:-n_outliers]
        predict_labels = predict_labels[:-n_outliers]
        f_adj = f_adj[:-n_outliers,:][:,:-n_outliers]

    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, adj, f1_macro, f1_micro = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj, f1_macro, f1_micro

def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num, A=None):
    f_adj = np.matmul(z, np.transpose(z))
    if A is not None:
        print("A→S")
        f_adj = f_adj * A
    cosine = f_adj
    cosine = cosine.reshape([-1,])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1-lower_treshold) * len(cosine))
    
    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
    
    return np.array(pos_inds), np.array(neg_inds)

def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

def gae_for(args):

    print("Using {} dataset".format(args.dataset))

    adj, features, true_labels = load_data(args.dataset)
    print("adj.shape=", adj.shape)
    print("features.shape=", features.shape)
    print("true_labels.shape=", true_labels.shape)

    # TODO 特征工程PCA
    # if(args.dataset == "dblp"):
    #     args.n_components = 256
    # pca = PCA(n_components=args.n_components, svd_solver='full')
    # features = pca.fit_transform(features)
    # print("PCA后,feature.shape=", features.shape)

    n_nodes, feat_dim = features.shape
    dims = [feat_dim] + args.dims

    [n_clusters, n_outliers] = cal_n_cluster(args.dataset)
    print("n_clusters=", n_clusters, "n_outliers=", n_outliers)
    Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    
    layers = args.linlayers
    # Store original adjacency matrix (without diagonal entries) for later

    # type = csr
    # diagonal()用于返回数组(也可称为矩阵)的对角线元素 [np.newaxis, :]给第0维增加维度
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # 只保留了有“1”的，不会改变shape
    adj.eliminate_zeros()

    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    
    n = adj.shape[0]

    # 构造拉普拉斯平滑器(多层)
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    # 存放平滑后的特征X'
    sm_fea_s = sp.csr_matrix(features).toarray()
    
    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    adj_1st = (adj + sp.eye(n)).toarray()
    A = None
    # # TODO 融一阶/二阶矩阵的正样本
    # t = 1
    # A = adj_1st
    # A = sum([np.linalg.matrix_power(A, i) for i in range(1, t + 1)]) / t
    # print("相似度计算,A-t:", t)

    # TODO
    # ValueError: array must not contain infs or NaNs
    # PCA存在的问题：https://stackoverflow.com/questions/41230558/pca-in-sklearn-valueerror-array-must-not-contain-infs-or-nans/42764378#42764378
    # db, best_acc, best_nmi, best_adj, best_f1_macro, best_f1_micro = clustering(Cluster, sm_fea_s, true_labels, n_outliers)
    # acc_list = [best_acc]
    # nmi_list = [best_nmi]
    # ari_list = [best_adj]
    # f1_list = [best_f1_macro]
    #
    # best_cl = db
    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    adj_label = torch.FloatTensor(adj_1st)
    
    model = LinTrans(layers, dims)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_label = adj_label.reshape([-1,])

    if args.cuda is True:
        model.cuda()
        # model = nn.DataParallel(model).cuda()
        inx = sm_fea_s.cuda()
        adj_label = adj_label.cuda()

    pos_num = len(adj.indices)
    neg_num = n_nodes*n_nodes-pos_num

    up_eta = (args.upth_ed - args.upth_st) / (args.epochs/args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs/args.upd)

    pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num, neg_num, A)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    bs = min(args.bs, len(pos_inds))
    length = len(pos_inds)
    
    pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
    print('Start Training...')
    for epoch in tqdm(range(args.epochs)):
        
        st, ed = 0, bs
        batch_num = 0
        model.train()
        length = len(pos_inds)
        
        while ( ed <= length ):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed-st)).cuda()
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()
            optimizer.zero_grad()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx = model(x)
            zy = model(y)
            batch_label = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).cuda()
            batch_pred = model.dcs(zx, zy)
            loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed-st)
            
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            
            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs

            
        if (epoch + 1) % args.upd == 0:
            model.eval()
            mu = model(inx)
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num, A)
            bs = min(args.bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
            
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))
            
            db, acc, nmi, adjscore, f1_macro, f1_micro = clustering(Cluster, hidden_emb, true_labels, n_outliers)
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(adjscore)
            f1_list.append(f1_macro)

            # if db >= best_cl:
            #     best_cl = db
            #     best_acc = acc
            #     best_nmi = nmi
            #     best_adj = adjscore
            #     best_f1_macro = f1_macro
            #     best_f1_micro = f1_micro
            
        
    tqdm.write("Optimization Finished!")
    
    acc_arr = np.array(acc_list)
    nmi_arr = np.array(nmi_list)
    ari_arr = np.array(ari_list)
    f1_arr = np.array(f1_list)

    return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr)
    

if __name__ == '__main__':

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        mean_acc, mean_nmi, mean_ari, mean_f1 = gae_for(args)

        final_acc.append(mean_acc)
        final_nmi.append(mean_nmi)
        final_ari.append(mean_ari)
        final_f1.append(mean_f1)

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)

    # print("acc:", acc_arr)
    # print("nmi:", nmi_arr)
    # print("ari:", ari_arr)
    # print("f1:", f1_arr)
    print(args)

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