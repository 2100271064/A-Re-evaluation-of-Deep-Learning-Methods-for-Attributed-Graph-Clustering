#python3

import numpy as np
import argparse
from tqdm import tqdm
from sklearn.datasets import load_wine
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import normalized_mutual_info_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm', help='Dataset to use')
parser.add_argument('-l', '--layers', nargs='+', type=int, default=[128, 64, 128], help='Sparsity Penalty Parameter')
parser.add_argument('-b', '--beta', type=float, default=0.01, help='Sparsity Penalty Parameter')
parser.add_argument('-p', '--rho', type=float, default=0.5, help='Prior rho')
# TODO 原来是0.01
parser.add_argument('-lr', type=float, default=0.0001, help='Learning Rate')
# TODO 原来是200
parser.add_argument('-epoch', type=int, default=400, help='Number of Training Epochs')
parser.add_argument('-device', type=str, default='gpu', help='Train on GPU or CPU')
parser.add_argument('--gpu', type=str, default='2')
args = parser.parse_args()

# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from torch import nn, optim
import torch
from sklearn.decomposition import PCA
from mymodel import GraphEncoder
from utils import *
from evaluation import eva

device = torch.device('cuda' if args.device == 'gpu' else 'cpu')


def main(model, X_train, Y, n_outliers):

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    with tqdm(total=args.epoch) as tq:
        for epoch in range(1, args.epoch + 1):
            optimizer.zero_grad()
            X_hat = model(X_train)
            loss = model.loss(X_hat, X_train, args.beta, args.rho)
            print('{} loss: {}'.format(epoch, loss))
            # TODO 指标
            # nmi = normalized_mutual_info_score(model.get_cluster(), Y, average_method='arithmetic')
            acc, nmi, ari, f1 = eva(Y, n_outliers, model.get_cluster(), epoch)
            if acc != -1: # 也可以不加，因为使用Kmeans
                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)
                f1_list.append(f1)

            loss.backward()
            optimizer.step()

            # tq.set_postfix(loss='{:.3f}'.format(loss), nmi='{:.3f}'.format(nmi))
            # tq.update()
        # print(model.get_cluster())
        print("Optimization Finished!")
        acc_arr = np.array(acc_list)
        nmi_arr = np.array(nmi_list)
        ari_arr = np.array(ari_list)
        f1_arr = np.array(f1_list)

        return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr)


if __name__ == '__main__':

    ###  model definition
    # if args.dataset.lower() == 'wine':
    #     data = load_wine()
    # else:
    #     raise Exception('Invalid dataset specified')

    # TODO dataset
    # X、Y：ndarray
    # X = data.data
    # Y = data.target
    # k = len(np.unique(Y))
    X, Y = load_data(args.dataset)
    [k, n_outliers] = cal_n_cluster(args.dataset)

    # TODO 特征工程
    if args.dataset != "pubmed":
        n_components = 500
        if (args.dataset == "dblp"):
            n_components = 256
        pca = PCA(n_components=n_components, svd_solver='full')
        X = pca.fit_transform(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Obtain Similarity matrix
    S = cosine_similarity(X, X)

    D = np.diag(1.0 / np.sqrt(S.sum(axis=1)))
    X_train = torch.tensor(D.dot(S).dot(D)).float().to(device)

    print(len(X_train))
    layers = [len(X_train)] + args.layers + [len(X_train)]

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))


        model = GraphEncoder(layers, k).to(device)


        mean_acc, mean_nmi, mean_ari, mean_f1 = main(model, X_train, Y, n_outliers)

        final_acc.append(mean_acc)
        final_nmi.append(mean_nmi)
        final_ari.append(mean_ari)
        final_f1.append(mean_f1)

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)

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

