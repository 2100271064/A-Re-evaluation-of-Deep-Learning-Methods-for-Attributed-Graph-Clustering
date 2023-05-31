'''
Author: xinying Lai
Date: 2022-09-07 21:23:19
LastEditTime: 2022-09-13 18:30:04
Description: Do not edit
'''
import opt
# 一定要在torch前面
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu

from train_mod import *
from DCRN import DCRN

if __name__ == '__main__':
    # setup
    # PUBMED and CORAFULL训练时可能会超出内存，需要用另一个代码
    # 分批训练的项目：https://drive.google.com/file/d/185GLObsQQL3Y-dQ2aIin5YrXuA-dgpnU/view?usp=sharing(google云端硬盘)
    
    # 将超参数设置好
    setup()

    # data pre-precessing: X, y, A, A_norm, Ad
    X, y, A = load_graph_data(opt.args.name, show_details=False)
    A_norm = normalize_adj(A, self_loop=True, symmetry=False)
    Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)

    # to torch tensor
    X = numpy_to_torch(X).to(opt.args.device)
    A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    Ad = numpy_to_torch(Ad).to(opt.args.device)


    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        # Dual Correlation Reduction Network
        model = DCRN(n_node=X.shape[0]).to(opt.args.device)

        [n_clusters, n_outliers] = cal_n_cluster(opt.args.name)
       # deep graph clustering
        mean_acc, mean_nmi, mean_ari, mean_f1 = train(model, X, y, A, A_norm, Ad, n_clusters, n_outliers)

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
    print('final_acc: {}, fianl_var_acc: {}'.format(value, var))
    print('final_acc: {:.4f}, fianl_var_acc: {:.2f}'.format(value, var))

    value = np.mean(nmi_arr)
    var = np.var(nmi_arr)
    print('final_nmi: {}, final_var_nmi: {}'.format(value, var))
    print('final_nmi: {:.4f}, final_var_nmi: {:.2f}'.format(value, var))

    value = np.mean(ari_arr)
    var = np.var(ari_arr)
    print('final_ari: {}, final_var_ari: {}'.format(value, var))
    print('final_ari: {:.4f}, final_var_ari: {:.2f}'.format(value, var))

    value = np.mean(f1_arr)
    var = np.var(f1_arr)
    print('final_f1: {}, final_var_f1: {}'.format(value, var))
    print('final_f1: {:.4f}, final_var_f1: {:.2f}'.format(value, var))
