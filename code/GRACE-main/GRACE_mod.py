import scipy.io as sio
import time
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
# from data import data
import argparse, os,sys,inspect
# from Laplacian_HGCN import Laplacian # for non-linear laplacian
from sklearn.preprocessing import normalize
import random
# from sknetwork.clustering import Louvain, BiLouvain, modularity
import networkx as nx
from networkx.readwrite import json_graph
import json
from sklearn.decomposition import PCA

p = argparse.ArgumentParser(description='Choose Parameter for Filter interpolation')
# 找数据集路径
p.add_argument('--data_type', type=str, default='cocitation', help='data name (coauthorship/cocitation)') # 我们用不到
p.add_argument('--dataset', type=str, default='acm', help='dataset name '
               '(e.g.: cora/dblp/acm for coauthorship, cora/citeseer/pubmed for cocitation)')
p.add_argument('--graph_type', type=str, default='Undirected', help='graph type ' # 我们用不到
                '(e.g.: Hypergraph, Heterogeneous, Multi-relational, Undirected, Directed)')
# 参数1
p.add_argument('--tol', type=float, default=0.15, help='tolerance')
# 参数2
p.add_argument('--power', type=int, default=40, help='order-k')
p.add_argument('--gpu', type=int, default=None, help='gpu number to use')
p.add_argument('--cuda', type=bool, default=False, help='cuda for gpu')
p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
p.add_argument('--mediators', type=int, default=1, help='use Mediators for Laplacian from FastGCN') # 我们用不到
# 参数3
p.add_argument('--normalize', type=str, default='None', help='Use l2 or l1 norm or None')
# 参数4
p.add_argument('--alpha', type=float, default=0.2, help='balance parameter')
p.add_argument('--beta', nargs='+', default=[0.7, 0.3, 0.3],  action='append', help='laplacian weights') # 我们用不到
# for beta, acm alpha 0.7*PAP 0.3*PLP [0.7,0.3]  # imdb [0.5,0.5] #dblp [0.3,0.3,0.3]
p.add_argument('--num_runs', type=int, default=2, help='num_runs') # 运行次数
p.add_argument('--loop', type=bool, default=True, help='num_runs')
p.add_argument('--lap_type', type=str, default='sym', help='sym, rw')
p.add_argument('--max_tol_count', type=int, default=2,
               help='number of times tol is met befor breaking') # 满足tol的次数，可以结束

# p.add_argument('-f') # for jupyter default
args = p.parse_args()


# def preprocess_adj_hyper(H, variable_weight=False):
#     """
#     calculate G from hypgraph incidence matrix H
#     :param H: hypergraph incidence matrix H
#     :param variable_weight: whether the weight of hyperedge is variable
#     :return: G
#     """
#     H = np.array(H)
#     n_edge = H.shape[1]
#
#     # the weight of the hyperedge
#     W = np.ones(n_edge)
#
#     # the degree of the node
#     DV = np.sum(H * W, axis=1)
#     degree_mat = np.mat(np.diag(DV))
#
#     # the degree of the hyperedge
#     DE = np.sum(H, axis=0)
#
#     invDE = np.mat(np.diag(np.power(DE, -1)))
#     DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     DV2[np.isinf(DV2)] = 0.
#     W = np.mat(np.diag(W))
#     H = np.mat(H)
#     HT = H.T
#
#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#
#         G = DV2.dot(H.dot(W.dot(invDE.dot(HT.dot(DV2)))))
#         I = sp.eye(G.shape[0]).toarray()
#         L = I - G
#
#         return L, H.dot(W.dot(HT)) - degree_mat, degree_mat


def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)

    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count == 0] = 1

    mean = onehot.dot(feature) / count
    a2 = (onehot.dot(feature * feature) / count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2 * mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist, inter_dist


# def dist(prelabel, feature):
#     k = len(np.unique(prelabel))
#     intra_dist = 0
#
#     for i in range(k):
#         Data_i = feature[np.where(prelabel == i)]
#         Dis = euclidean_distances(Data_i, Data_i)
#         n_i = Data_i.shape[0]
#         if n_i == 0 or n_i == 1:
#             intra_dist = intra_dist
#         else:
#             intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))
#
#     return intra_dist


def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)

        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


# def get_norm_adj_all(list_of_adj, type='sym', loop=True):
#
#     for i in range(len(list_of_adj)):
#         list_of_adj[i] = preprocess_adj(list_of_adj[i], type, loop)
#     return list_of_adj


# def Incidence_mat(features, Hypergraphs):
#     print("creating incidence matrix")
#     Incidence = np.zeros(shape=(features.shape[0], len(Hypergraphs)))
#     for edgei, (k, v) in enumerate(Hypergraphs.items()):
#         for i in v:
#             Incidence[i][edgei] = 1
#     return Incidence


def Build_filters(grap_num_nodes, graph_Structure, graph_features, mediators=args.mediators,
                  alpha=args.alpha, graph_type=args.graph_type, beta=args.beta, type=args.lap_type,
                  loops=args.loop, seed = args.seeds
                  ):

    np.random.seed(seed)
    random.seed(seed)

    # if graph_type == 'Hypergraph':
    #     graph_filter = Laplacian(grap_num_nodes, graph_Structure, graph_features, mediators)
    #     graph_filter = (1 - alpha) * sp.eye(graph_filter.shape[0]) + alpha * graph_filter
    #
    # elif graph_type in ['Heterogeneous', 'Multi-Relational']:
    #     graph_Structure = get_norm_adj_all(graph_Structure, type=type, loop=loops)
    #     graph_filter = 0
    #     for i in range(len(graph_Structure)):
    #         graph_filter += beta[i] * (sp.eye(graph_Structure[i].shape[0]) - graph_Structure[i])  ###
    #
    #     graph_filter = sp.eye(graph_filter.shape[0]) - (alpha) * graph_filter

    # ----------- we focus -------------
    # elif graph_type == 'Undirected':
    graph_filter = preprocess_adj(graph_Structure, loop=loops, type=type)
    graph_filter = (1 - alpha) * sp.eye(graph_filter.shape[0]) + alpha * graph_filter

    # elif graph_type == 'Directed':
    #     graph_filter = graph_Structure + graph_Structure.T
    #     graph_filter = preprocess_adj(graph_filter, loop=loops, type=type)
    #     graph_filter = (1 - alpha) * sp.eye(graph_filter.shape[0]) + alpha * graph_filter
    #
    # else:
    #     print(f'======== graph_type: {graph_type} has not been implemented ========')

    return graph_filter


# def load_npz_dataset(file_name):
#     """Load a graph from a Numpy binary file.
#     Parameters
#     ----------
#     file_name : str
#         Name of the file to load.
#     Returns
#     -------
#     graph : dict
#         Dictionary that contains:
#             * 'W' : The adjacency matrix in sparse matrix format
#             * 'fea' : The attribute matrix in sparse matrix format
#             * 'gnd' : The ground truth class labels
#             * Further dictionaries mapping node, class and attribute IDs
#     """
#     if not file_name.endswith('.npz'):
#         file_name += '.npz'
#     with np.load(file_name, allow_pickle=True) as loader:
#         loader = dict(loader)
#         edge_index = loader['adj_indices'].copy()
#         A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
#                            loader['adj_indptr']), shape=loader['adj_shape'])
#
#         X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
#                            loader['attr_indptr']), shape=loader['attr_shape'])
#
#         z = loader.get('labels')
#
#         graph = {
#             'W': A,
#             'fea': X,
#             'gnd': z
#         }
#
#         idx_to_node = loader.get('idx_to_node')
#         if idx_to_node:
#             idx_to_node = idx_to_node.tolist()
#             graph['idx_to_node'] = idx_to_node
#
#         idx_to_attr = loader.get('idx_to_attr')
#         if idx_to_attr:
#             idx_to_attr = idx_to_attr.tolist()
#             graph['idx_to_attr'] = idx_to_attr
#
#         idx_to_class = loader.get('idx_to_class')
#         if idx_to_class:
#             idx_to_class = idx_to_class.tolist()
#             graph['idx_to_class'] = idx_to_class
#
#         return graph

# TODO ----------- we focus -------------
def load_simple(type=args.graph_type, dataset=args.dataset):

    root_path = r"/home/laixy/AHG/dataset/{0}".format(dataset)

    # 抽取需要的
    node = np.loadtxt(root_path + r"/node.txt", dtype=int)

    if (dataset != "us" and dataset != "uk"):
        # 全部顶点的特征矩阵
        feature = np.loadtxt(root_path + r"/feature.txt", dtype=float)
        feature = feature[node]
        # 全部顶点的邻接矩阵
        adj = np.loadtxt(root_path + r"/adj.txt", dtype=int)
        adj = adj[node]
        adj = adj[:, node]
        adj = sp.coo_matrix(adj)
    else:
        feature = np.load(root_path + r"/feat192.npy")
        feature = feature[node]
        nx_graph = json_graph.node_link_graph(json.load(open(root_path + r"/(renumberated)graph.json")))
        adj = nx.to_scipy_sparse_matrix(nx_graph, format='coo')

    labels = np.loadtxt(root_path + r"/label.txt", dtype=int)

    def cal_n_cluster(dataset):
        # return [cluster_num, outliers_num]
        if dataset == 'cora':
            args.normalize = 'l2'
            return [18, 7]
        elif dataset == 'citeseer':
            args.normalize = 'None'
            return [12, 1]
        elif dataset == 'pubmed':
            args.normalize = 'l1'
            return [26, 0]
        elif dataset == 'wiki':
            args.normalize = 'l1'
            return [17, 5]
        elif dataset == 'acm':
            args.normalize = 'l2'
            return [12, 0]
        elif dataset == 'dblp':
            args.normalize = 'l2'
            return [17, 0]
        elif dataset == 'uk' or dataset == 'us':
            return [20, 0]
    [n_clusters, n_outliers] = cal_n_cluster(dataset)

    return adj, feature, labels, feature.shape[0], n_clusters, n_outliers


# def load_acm():
#     '''From https://gitlab.cs.univie.ac.at/yllis19cs/spectralmixpublic'''
#     feature = 'feature'
#     path = 'data/acm/'
#     feature = sio.loadmat('{}{}.mat'.format(path, feature))
#     adj1 = 'PAP'
#     adj2 = 'PLP'
#
#     adj1 = sio.loadmat('{}{}.mat'.format(path, adj1))
#     adj2 = sio.loadmat('{}{}.mat'.format(path, adj2))
#
#     adj1 = sp.coo_matrix(adj1['PAP'])
#     adj2 = sp.coo_matrix(adj2['PLP'])
#
#     feature = feature['feature']
#
#     list_of_adj = [adj1, adj2]
#
#     lines = 'ground_truth'
#     gt = []
#     with open('{}{}.txt'.format(path, lines)) as f:
#         lines = f.readlines()
#     for line in lines:
#         gt.append(int(line))
#
#     gt = np.array(gt)
#
#     return list_of_adj, feature, gt


# def load_imdb():
#     '''From https://gitlab.cs.univie.ac.at/yllis19cs/spectralmixpublic'''
#     path = 'data/imdb/'
#     adj1 = 'imdb'
#     ids = 'ids'
#
#     adj1 = sio.loadmat('{}{}.mat'.format(path, adj1))
#     ids = sio.loadmat('{}{}.mat'.format(path, ids))
#
#     list_of_adj = []
#     list_of_adj.append(adj1['MDM'])
#     list_of_adj.append(adj1['MAM'])
#
#     feature = adj1['feature']
#
#     lines = 'ground_truth'
#     gt = []
#     count = 0
#     with open('{}{}.txt'.format(path, lines)) as f:
#         lines = f.readlines()
#
#     for line in lines:
#         count += 1
#         # print(line, count)
#         if isinstance(line, str):
#             gt.append(int(line))
#         else:
#             gt.append(line)
#     gt = np.array(gt)
#
#     return list_of_adj, feature, gt


# def load_dblp():
#     '''From: https://github.com/liun-online/HeCo/tree/f1652585679dd50315044f77c56cafaa3942c294/code/utils'''
#
#     path = 'data/dblpAttributed/'
#     adj1 = 'apa'
#     adj2 = 'apcpa'
#     adj3 = 'aptpa'
#     feature = 'a_feat'
#     gt = 'labels'
#
#     adj1 = sp.load_npz('{}{}.npz'.format(path, adj1))
#     adj2 = sp.load_npz('{}{}.npz'.format(path, adj2))
#     adj3 = sp.load_npz('{}{}.npz'.format(path, adj3))
#
#     print(np.count_nonzero(adj1.toarray()) + np.count_nonzero(adj2.toarray()) + np.count_nonzero(adj3.toarray()))
#     list_of_adj = [adj1, adj2, adj3]
#
#     feature = sp.load_npz('{}{}.npz'.format(path, feature))
#
#     gt = np.load('{}{}.npy'.format(path, gt)).astype('int32')
#
#     return list_of_adj, feature, gt


# def load_hete_or_MR(dataset=args.dataset):
#     if args.dataset == 'acm':
#         list_of_adj, feature, labels = load_acm()
#
#     if args.dataset == 'imdb':
#         list_of_adj, feature, labels = load_imdb()
#
#     elif args.dataset == 'dblp':
#         list_of_adj, feature, labels = load_dblp()
#         # print('not yet implemented')
#         # exit(0)
#
#     if sp.issparse(feature):
#         feature = feature.todense()
#
#     return list_of_adj, feature, labels, feature.shape[0]


def load_dataset(dataset=args.dataset, graph_type=args.graph_type, data_type=args.data_type):

    # if graph_type == 'Hypergraph':
    #     dataset = data.load(args.data_type, args.dataset)
    #     graph_structure = dataset['hypergraph']
    #     features = dataset['features']
    #     labels = dataset['labels']
    #     num_nodes = dataset['n']
    #     num_hyperedges = dataset['e']
    #     labels = np.asarray(np.argmax(labels, axis=1))
    #     return graph_structure, features, labels, num_nodes
    #
    # elif graph_type in ['Heterogeneous', 'Multi-Relational']:
    #     return  load_hete_or_MR(dataset=dataset)

    # ----------- we focus -------------
    # elif graph_type in ['Undirected', 'Directed']:
    return load_simple(type=graph_type, dataset=dataset)

    # else:
    #     print('======== graph_type : {} Not Implemented ========')
    #     exit(0)

def main(features, adj_normalized, labels, k, n_outliers):

    max_count = args.max_tol_count

    intra_list = []
    inter_list = []
    acc_list = []
    stdacc_list = []
    f1_list = []
    stdf1_list =[]
    nmi_list = []
    stdnmi_list = []
    ncut_list = []
    precision_list = []
    adj_score_list = []
    recall_macro_list = []
    # modularity_list = []

    intra_list.append(10000000)
    inter_list.append(10000000)
    rep = 1
    count = 0

    t = time.time()

    # args.power = 40
    for p in range(1, args.power + 1):
        seed = args.seeds
        np.random.seed(seed)
        random.seed(seed)

        e = p - 1

        IntraD = np.zeros(rep)
        InterD = np.zeros(rep)
        # Ncut = np.zeros(rep)
        ac = np.zeros(rep)
        nm = np.zeros(rep)
        f1 = np.zeros(rep)
        pre = np.zeros(rep)
        rec = np.zeros(rep)
        adj_s = np.zeros(rep)
        # mod = np.zeros(rep)

        features = adj_normalized.dot(features)

        # if args.graph_type == 'Hypergraph':
        #     adj_normalized = Laplacian(num_nodes, structure, features, args.mediators)
        #     adj_normalized = (1 - alpha) * sp.eye(adj_normalized.shape[0]) + alpha * adj_normalized

        if args.normalize == 'l2':
            Trick = normalize(features, norm='l2', axis=1) ## good

        elif args.normalize == 'l1':
            Trick = normalize(features, norm='l1', axis=1)

        else:
            Trick = features

        # sp.linalg.svds是SciPy库中用于计算奇异值分解(SVD)的函数。SVD是一种常用的矩阵分解方法，可以将一个矩阵分解成三个矩阵的乘积，
        # 即$A=U \Sigma V^T$，其中$U$和$V$是正交矩阵，$\Sigma$是对角矩阵。
        # SVD常用于数据压缩、降维和噪声滤除等领域。
        # sp.linalg.svds函数可用于计算矩阵的前k个奇异值和对应的左右奇异向量。
        u, s, v = sp.linalg.svds(Trick, k=k, which='LM')

        for i in range(rep):

            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=args.seeds).fit(u)
            predict_labels = kmeans.predict(u)

            # Ncut[i] = Normalized_cut(predict_labels, Lap, degree_mat
            # 判断结束的两个指标
            IntraD[i], InterD[i] = square_dist(predict_labels, features)
            #intraD[i] = dist(predict_labels, features)

            # 处理n_outliers
            cm = clustering_metrics(labels, predict_labels, n_outliers)
            # 聚类结果指标 acc, nmi, f1, _, ari, _
            ac[i], nm[i], f1[i], pre[i], adj_s[i], rec[i] = cm.evaluationClusterModelFromLabel()
            acc_value = ac[i]
            nmi_value = nm[i]
            ari_value = adj_s[i]
            f1_value = f1[i]
            # mod[i] = modularity(predict_labels, adj)

        intramean = np.mean(IntraD)
        intermean = np.mean(InterD)
        # ncut_mean = np.mean(Ncut)
        acc_means = np.mean(ac)
        acc_stds = np.std(ac)
        nmi_means = np.mean(nm)
        nmi_stds = np.std(nm)
        f1_means = np.mean(f1)
        f1_stds = np.std(f1)
        # mod_means = np.mean(mod)
        pre_mean = np.mean(pre)
        rec_mean = np.mean(rec)
        adj_smean = np.mean(adj_s)

        # modularity_list.append(mod_means)
        # ncut_list.append(ncut_mean)
        intra_list.append(intramean)
        inter_list.append(intermean)
        acc_list.append(acc_means)
        stdacc_list.append(acc_stds)
        nmi_list.append(nmi_means)
        stdnmi_list.append(nmi_stds)
        f1_list.append(f1_means)
        stdf1_list.append(f1_stds)
        precision_list.append(pre_mean)
        recall_macro_list.append(rec_mean)
        adj_score_list.append(adj_smean)

        # if args.graph_type == 'Hypergraph':
        #     print('dataset: {}_{}, power: {}, ac: {}, f1: {}, nm: {}, intraD: {}, InterD: {}, pre: {}, rec: {}, adj_score: {}'.format(args.dataset, args.data_type, p, acc_means, f1_means, nmi_means, intramean, intermean, pre_mean, rec_mean, adj_smean))
        # else:
        print('dataset: {}_{}, power: {}, ac: {}, f1: {}, nm: {}, intraD: {}, InterD: {}, pre: {}, rec: {}, adj_score: {}'.format(args.dataset, args.graph_type, p, acc_means, f1_means, nmi_means, intramean, intermean, pre_mean, rec_mean, adj_smean))


        # 满足条件
        if intra_list[e]- intra_list[p] <= args.tol:
            count += 1
            print('count: {}'.format(count))

        # 当多少个满足条件时，就跳出
        if count >= max_count:
            print('=====================Breaking As Condition Met================')
            print('power: {}, ac: {}, f1: {}, nm: {}, intraD: {}, InterD: {}, pre: {}, rec: {}, adj_score: {}'.format(p, acc_list[e], f1_list[e], nmi_list[e], intra_list[e], inter_list[e], precision_list[e], recall_macro_list[e], adj_score_list[e]))
            t = time.time() - t
            print('time taken: {}'.format(t))
            acc_value = acc_list[e]
            nmi_value = nmi_list[e]
            ari_value = adj_score_list[e]
            f1_value = f1_list[e]
            break

    return acc_value, nmi_value, ari_value, f1_value

if __name__ == '__main__':

    structure, features, labels, num_nodes, n_clusters, n_outliers = load_dataset(dataset=args.dataset,
                                                          graph_type=args.graph_type,
                                                          data_type=args.data_type)


    # TODO 特征工程
    if args.dataset != "pubmed":
        n_components = 500
        if (args.dataset == "dblp"):
            n_components = 256
        pca = PCA(n_components=n_components, svd_solver='full')
        features = pca.fit_transform(features)

    k = n_clusters
    # if args.graph_type == 'Hypergraph':
    #     print(f'---------------- {args.dataset}_{args.data_type} loaded ----------------')
    #     alpha = args.alpha

    # else:
    print(f'---------------- {args.dataset}_{args.graph_type} loaded ----------------')
    print(f'number of nodes: {num_nodes}')
    print(f'number distinct labels: {k}')
    print(f'number of user specified clusters: {k}')
    print(f'will use: {args.normalize} as normalization for similarity matrix')

    adj_normalized = Build_filters(num_nodes, structure, features, mediators=args.mediators,
                                   alpha=args.alpha, graph_type=args.graph_type, beta=args.beta, type=args.lap_type,
                                   loops=args.loop, seed=args.seeds)

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))


        mean_acc, mean_nmi, mean_ari, mean_f1 = main(features, adj_normalized, labels, k, n_outliers)

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
