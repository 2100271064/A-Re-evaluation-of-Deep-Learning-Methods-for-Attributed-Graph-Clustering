import opt
# 一定要在torch前面
import os
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpu

from ATT_AE import *
from GCN import *
from Combine import *
from Merge import *
from Merge_ATT import *
# from Combine2 import *
# from Merge2 import *
# from Merge_ATT2 import *
# from AGE_GCN import *
# from AGE_GAT import *
# TODO
# from train_AGE import *
from train import *
from train_pubmed import *
# TODO
# from utils_AGE import *
from utils import *

if __name__ == '__main__':

    # 将超参数设置好
    # TODO
    setup()

    print("Using {} dataset".format(opt.args.name))

    # TODO: 原本的adj, adj_(from DFCN), adj_label, features, labels (train_AGE)
    adj, adj_, adj_label, feature, label = load_data(opt.args.name)
    # adj, feature, label = load_data(opt.args.name)
    print("adj.shape=", adj.shape)
    print("features.shape=", feature.shape)
    print("true_labels.shape=", label.shape)

    # TODO 特征工程PCA
    if (opt.args.name == "dblp"):
        opt.args.n_components = 256
    # pubmed本来就是500，无需进行PCA处理
    if (opt.args.name != "pubmed"):
        pca = PCA(n_components=opt.args.n_components, svd_solver='full')
        feature = pca.fit_transform(feature)
        print("PCA后,feature.shape=", feature.shape)
    else:
        print("{}没有进行PCA处理".format(opt.args.name))


    opt.args.input_dim = feature.shape[1]


    # TODO AGE相关
    adj_AGE, sm_fea_s = laplacian_smoothing(adj, feature)
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    inx = sm_fea_s.to(opt.args.device)
    # TODO GCN(DFCN相关 Train_AGE)
    adj_ = adj_.to(opt.args.device)

    # TODO 非AGE相关
    # adj_I = adj + sp.eye(adj.shape[0])
    # adj_normalized = normalize(adj_I)
    # adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # feature = feature.to(opt.args.device)
    # adj_normalized_dense = adj_normalized.to_dense()
    # adj_normalized = adj_normalized_dense.to(opt.args.device)
    adj_label = adj_label.to(opt.args.device)

    # ATT-AE(增强)
    # t = 2
    # adj_normalized_dense_numpy = adj_normalized_dense.data.cpu().numpy()
    # tran_prob = normalize(adj_normalized_dense_numpy, norm="l1", axis=0)
    # M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    # M = torch.Tensor(M_numpy).to(opt.args.device)

    run_round = 10
    final_acc = []
    final_nmi = []
    final_ari = []
    final_f1 = []
    valid_epoch_num_list = []
    for i in range(run_round):
        print('----------------------round_{0}-----------------------------'.format(i))

        # TODO
        # model = ATT_AE(num_features=opt.args.input_dim, hidden_size=opt.args.hidden1_dim,
        #           embedding_size=opt.args.hidden2_dim, alpha=opt.args.alpha, num_clusters=opt.args.n_clusters).to(opt.args.device)
        # model = GCN(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
        #             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
        #             n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1).to(opt.args.device)
        # TODO n_node=None(pubmed)/ n_node=feature.shape[0](other)
        # model = Combine(n_enc_1=opt.args.n_enc_1, n_enc_2=opt.args.n_enc_2, n_enc_3=opt.args.n_enc_3,
        #             n_dec_1=opt.args.n_dec_1, n_dec_2=opt.args.n_dec_2, n_dec_3=opt.args.n_dec_3,
        #             n_z=opt.args.n_z, n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
        #             n_node=None, device=opt.args.device).to(opt.args.device)
        model = Merge(n_enc_1=opt.args.n_enc_1, n_enc_2=opt.args.n_enc_2, n_enc_3=opt.args.n_enc_3,
                    n_dec_1=opt.args.n_dec_1, n_dec_2=opt.args.n_dec_2, n_dec_3=opt.args.n_dec_3, n_z=opt.args.n_z,
                    n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, alpha=opt.args.alpha, v=1).to(opt.args.device)
        # model = Merge_ATT(n_enc_1=opt.args.n_enc_1, n_enc_2=opt.args.n_enc_2, n_enc_3=opt.args.n_enc_3,
        #               n_dec_1=opt.args.n_dec_1, n_dec_2=opt.args.n_dec_2, n_dec_3=opt.args.n_dec_3, n_z=opt.args.n_z,
        #               n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, alpha=opt.args.alpha, v=1).to(opt.args.device)
        # model = Combine2(n_enc_1=opt.args.n_enc_1, n_enc_2=opt.args.n_enc_2, n_enc_3=opt.args.n_enc_3,
        #             n_dec_1=opt.args.n_dec_1, n_dec_2=opt.args.n_dec_2, n_dec_3=opt.args.n_dec_3, n_z=opt.args.n_z, alpha=opt.args.alpha,
        #             n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1,
        #             n_node=feature.shape[0], device=opt.args.device).to(opt.args.device)
        # model = Merge2(n_enc_1=opt.args.n_enc_1, n_enc_2=opt.args.n_enc_2, n_enc_3=opt.args.n_enc_3,
        #             n_dec_1=opt.args.n_dec_1, n_dec_2=opt.args.n_dec_2, n_dec_3=opt.args.n_dec_3, n_z=opt.args.n_z, alpha=opt.args.alpha,
        #             n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1).to(opt.args.device)
        # model = Merge_ATT2(n_enc_1=opt.args.n_enc_1, n_enc_2=opt.args.n_enc_2, n_enc_3=opt.args.n_enc_3,
        #               n_dec_1=opt.args.n_dec_1, n_dec_2=opt.args.n_dec_2, n_dec_3=opt.args.n_dec_3, n_z=opt.args.n_z, alpha=opt.args.alpha,
        #               n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1).to(opt.args.device)
        # model = AGE_GCN(gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
        #          age_dim=opt.args.age_dim,
        #          n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1.0, n_node=adj.shape[0], device=opt.args.device).to(opt.args.device)
        # model = AGE_GAT(hidden1_dim=opt.args.hidden1_dim, hidden2_dim=opt.args.hidden2_dim,
        #                 age_dim=opt.args.age_dim, alpha=opt.args.alpha,
        #                 n_input=opt.args.input_dim, n_clusters=opt.args.n_clusters, v=1.0, n_node=adj.shape[0],
        #                 device=opt.args.device).to(opt.args.device)

        # TODO adj_/adj_normalized
        # mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = Train_AGE(opt.args.epoch, model, inx, adj_AGE, adj_, adj_label, label, opt.args.device)
        # TODO 更改函数名 train/train_pubmed
        mean_acc, mean_nmi, mean_ari, mean_f1, valid_epoch_num = train_pubmed(model, inx, adj, adj_, adj_label, label, opt.args.n_outliers, opt.args.device)

        final_acc.append(mean_acc)
        final_nmi.append(mean_nmi)
        final_ari.append(mean_ari)
        final_f1.append(mean_f1)
        valid_epoch_num_list.append(valid_epoch_num)

    # # TODO 有些loss需要关闭
    # save_path = r"../../../data/Laixy/model_test/"
    # # TODO
    # filename = "Merge_ATT2_Loss6"
    # # 保存pos_inds
    # # save_pos_inds(save_path, filename, pos_inds, n)
    # # 保存attention_alpha
    # save_attention_alpha(save_path, filename, attention_alpha)

    acc_arr = np.array(final_acc)
    nmi_arr = np.array(final_nmi)
    ari_arr = np.array(final_ari)
    f1_arr = np.array(final_f1)
    print("{} epoch × 10, 有效的epoch数：".format(opt.args.epoch), valid_epoch_num_list)

    value = np.mean(acc_arr)
    var = np.var(acc_arr)
    std = np.std(acc_arr)
    print('final_acc: {}, fianl_var_acc: {}, final_std_acc:{}'.format(value, var, std))
    print('final_acc: {:.4f}, fianl_var_acc: {:.2f}, final_std_acc:{:.2f}%'.format(value, var, std*100))

    value = np.mean(nmi_arr)
    var = np.var(nmi_arr)
    std = np.std(nmi_arr)
    print('final_nmi: {}, final_var_nmi: {}, final_std_nmi:{}'.format(value, var, std))
    print('final_nmi: {:.4f}, final_var_nmi: {:.2f}, final_std_nmi:{:.2f}%'.format(value, var, std*100))

    value = np.mean(ari_arr)
    var = np.var(ari_arr)
    std = np.std(ari_arr)
    print('final_ari: {}, final_var_ari: {}, final_std_ari:{}'.format(value, var, std))
    print('final_ari: {:.4f}, final_var_ari: {:.2f}, final_std_ari:{:.2f}%'.format(value, var, std*100))

    value = np.mean(f1_arr)
    var = np.var(f1_arr)
    std = np.std(f1_arr)
    print('final_f1: {}, final_var_f1: {}, final_std_f1:{}'.format(value, var, std))
    print('final_f1: {:.4f}, final_var_f1: {:.2f}, final_std_f1:{:.2f}%'.format(value, var, std*100))