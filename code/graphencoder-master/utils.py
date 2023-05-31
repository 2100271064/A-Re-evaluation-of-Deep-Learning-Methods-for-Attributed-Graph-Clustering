'''
Author: xinying Lai
Date: 2022-09-11 18:37:25
LastEditTime: 2022-09-11 18:39:17
Description: Do not edit
'''
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from networkx.readwrite import json_graph
import json


def load_data(dataset):

    root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)

    # 抽取需要的
    node = np.loadtxt(root_path + r"/node.txt", dtype=int)

    if (dataset != "us" and dataset != "uk"):
        # 全部顶点的特征矩阵
        features = np.loadtxt(root_path + r"/feature.txt", dtype=float)
        features = features[node]
    else:
        features = np.load(root_path + r"/feat192.npy")
        features = features[node]

    labels = np.loadtxt(root_path + r"/label.txt", dtype=int)

    return features, labels

def cal_n_cluster(dataset):
    # return [cluster_num, outliers_num]
    if dataset == 'cora':
        return [18, 7]
    elif dataset == 'citeseer':
        return [12, 1]
    elif dataset == 'pubmed':
        return [26, 0]
    elif dataset == 'wiki':
        return [17, 5]
    elif dataset == 'acm':
        return [12, 0]
    elif dataset == 'dblp':
        return [17, 0]
    elif dataset == 'uk' or dataset == 'us':
        return [20, 0]
