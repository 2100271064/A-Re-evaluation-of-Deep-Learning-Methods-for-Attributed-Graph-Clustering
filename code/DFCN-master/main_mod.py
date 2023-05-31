'''
Author: xinying Lai
Date: 2022-09-12 22:12:10
LastEditTime: 2022-09-13 18:29:59
Description: Do not edit
'''
import opt
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu
import torch
import numpy as np
from DFCN import DFCN
from utils_mod import setup_seed, cal_n_cluster
from sklearn.decomposition import PCA
from load_data_mod import LoadDataset, load_graph, construct_graph
from train_mod import Train
from train_mod_pubmed import Train_pubmed

setup_seed(opt.args.seed)

print("network setting…")

# TODO 原来是打开的
# if opt.args.name == "dblp":
#     opt.args.n_input = 50
# else:
#     opt.args.n_input = 100

[opt.args.n_clusters, n_outliers] = cal_n_cluster(opt.args.name)
opt.args.k = None


### cuda
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

### root
opt.args.data_path = '/home/laixy/AHG/dataset/{}/feature.txt'.format(opt.args.name)
opt.args.label_path = '/home/laixy/AHG/dataset/{}/label.txt'.format(opt.args.name)
# opt.args.graph_k_save_path = '/home/laixy/AHG/DFCN-master/graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
# opt.args.graph_save_path = '/home/laixy/AHG/DFCN-master/graph/{}_graph.txt'.format(opt.args.name)
opt.args.pre_model_save_path = '/home/laixy/AHG/data/DFCN-master/model_pre/{}_DFCN.pkl'.format(opt.args.name)
opt.args.final_model_save_path = '/home/laixy/AHG/data/DFCN-master/model_final/{}_DFCN.pkl'.format(opt.args.name)

### data pre-processing
print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))


x = np.loadtxt(opt.args.data_path, dtype=float)
vertex_arr = np.loadtxt('/home/laixy/AHG/dataset/{}/node.txt'.format(opt.args.name), dtype=int)
x = x[vertex_arr]
y = np.loadtxt(opt.args.label_path, dtype=int)

if (opt.args.name == "dblp"):
    opt.args.n_components = 256
pca = PCA(n_components=opt.args.n_components, svd_solver='full')
X_pca = pca.fit_transform(x)
# plot_pca_scatter(args.name, args.n_clusters, X_pca, y)

dataset = LoadDataset(X_pca)

# 里面已经有adj的vertex_arr筛选
adj = load_graph(opt.args.name).to(device)
data = torch.Tensor(dataset.x).to(device)
label = y


### training lr
print("Training on {}…".format(opt.args.name))

lr = 1e-4
# TODO 原来是打开的
# if opt.args.name == "acm":
#     lr = 5e-5
# else:
#     lr = 1e-4




run_round = 10
final_acc = []
final_nmi = []
final_ari = []
final_f1 = []
for i in range(run_round):
    print('----------------------round_{0}-----------------------------'.format(i))

    ###  model definition
    model = DFCN(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
                ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
                gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
                gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
                n_input=opt.args.n_components,
                n_z=opt.args.n_z,
                n_clusters=opt.args.n_clusters,
                v=opt.args.freedom_degree,
                # TODO 批处理
                n_node=None,
                # n_node=data.size()[0],
                 device=device).to(device)

    # TODO 批处理切换
    # mean_acc, mean_nmi, mean_ari, mean_f1 = \
    #     Train(opt.args.epoch, model, data, adj, label, n_outliers, lr, opt.args.pre_model_save_path, opt.args.final_model_save_path,
    #                         opt.args.n_clusters, opt.args.acc, opt.args.gamma_value, opt.args.lambda_value, device)
    mean_acc, mean_nmi, mean_ari, mean_f1 = \
        Train_pubmed(opt.args.epoch, model, data, adj, label, n_outliers, lr, opt.args.pre_model_save_path,
              opt.args.final_model_save_path,
              opt.args.n_clusters, opt.args.acc, opt.args.gamma_value, opt.args.lambda_value, device)
    
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


# print("ACC: {:.4f}".format(max(acc_reuslt)))
# print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
# print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
# print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
# print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])
