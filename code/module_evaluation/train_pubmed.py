import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import opt
from utils import *

import warnings
warnings.filterwarnings('ignore')

def train_pubmed(model, data, original_adj, adj, adj_label, label, n_outliers, device):

    print("Using train_pubmed method!")

    # TODO
    # loss：正负样本的标签生成
    # original_adj.eliminate_zeros()
    # # 初始化正负样本数量
    # pos_num = len(original_adj.indices)
    # n = original_adj.shape[0]
    # neg_num = n * n - pos_num
    # # 均等计算阈值增量
    # up_eta = (opt.args.upth_ed - opt.args.upth_st) / (opt.args.epoch / opt.args.upd)
    # low_eta = (opt.args.lowth_ed - opt.args.lowth_st) / (opt.args.epoch / opt.args.upd)
    # # 计算初始化时的正负样本下标，及并更新下次阈值
    # # TODO 没有融A的正样本
    # # pos_inds = update_similarity(normalize(data.cpu().numpy()), opt.args.upth_st, pos_num)
    # pos_inds, neg_inds = update_similarity(normalize(data.cpu().numpy()), opt.args.upth_st, opt.args.lowth_st, pos_num, neg_num, A=None)
    # # upth = update_threshold(opt.args.upth_st, up_eta)
    # upth, lowth = update_threshold(opt.args.upth_st, opt.args.lowth_st, up_eta, low_eta)
    # # TODO 融一阶/二阶矩阵的正样本
    # # t = 2
    # # A = (original_adj + sp.eye(adj.shape[0])).toarray()
    # # A = sum([np.linalg.matrix_power(A, i) for i in range(1, t + 1)]) / t
    # # print("相似度计算,A-t:", t)
    # # pos_inds = update_similarity(normalize(data.cpu().numpy()), opt.args.upth_st, pos_num, A)
    #
    # pos_inds_device = torch.LongTensor(pos_inds).to(device)
    # bs = min(opt.args.bs, len(pos_inds))

    acc_result = []
    nmi_result = []
    ari_result = []
    f1_result = []

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    with torch.no_grad():

        model_cpu = model.cpu()
        model_cpu.device = "cpu"
        # TODO Combine
        # model_cpu.a = model_cpu.a.cpu()
        # model_cpu.b = model_cpu.b.cpu()
        data = data.cpu()
        adj = adj.cpu()

        # ATT_AE
        # _, _, z, _, _, _ = model_cpu(data, adj)
        # GCN
        # _, _, z, _, _ = model_cpu(data, adj)
        # 组合模型3
        # _, _, _, _, _, _, _, _, z, _, _ = model_cpu(data, adj)
        # 组合模型4/5
        _, _, _, _, z, _, _ = model_cpu(data, adj)
        # Combine2
        # _, _, _, _, _, _, _, _, z, _, _, _ = model(data, adj)
        # Merge2 / Merge_ATT2
        # _, _, _, _, z, _, _, _ = model(data, adj)


    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    #y_pred_last = y_pred       #kmeans.cluster_centers_  从z中获得初始聚类中心
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(label, n_outliers, y_pred, 'init')

    for epoch in range(opt.args.epoch):

        # TODO
        # loss_A：正负样本的标签生成
        # model = model.to(device)
        # model.device = "cuda"
        # # TODO Combine
        # # model_cpu.a = model_cpu.a.cpu()
        # # model_cpu.b = model_cpu.b.cpu()
        # data = data.to(device)
        #
        # st, ed = 0, bs
        # batch_num = 0
        # model.train()
        # length = len(pos_inds)
        #
        # # 负样本是除去正样本的那些（共有n(n+1)/2-len(pos)个）
        # # sampled_neg = cal_neg_inds(pos_inds, n)
        # while (ed <= length):
        #     sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed-st))
        #     neg_inds_device = torch.LongTensor(sampled_neg).to(device)
        #     sampled_inds = torch.cat((pos_inds_device[st:ed], neg_inds_device), 0)
        #     # print("正样本数量：", len(pos_inds), "负样本数量：", len(sampled_neg))
        #     # print("n(n+1)/2=", (n*n+n)/2, "正负样本对(正样本有重复)=", len(pos_inds)+len(sampled_neg))
        #     optimizer.zero_grad()
        #     xind = sampled_inds // n
        #     yind = sampled_inds % n
        #     _, _, zx, _, _ = model(data[xind], sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj.to_dense()[xind, :][:, xind])).to(device))
        #     _, _, zy, _, _ = model(data[yind], sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj.to_dense()[yind, :][:, yind])).to(device))
        #     # zx = torch.index_select(z, 0, xind)
        #     # zy = torch.index_select(z, 0, yind)
        #     sample_label = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).to(device)
        #     sample_pred = (zx * zy).sum(1)
        #     loss = loss_function(preds=sample_pred,  labels=sample_label)
        #
        #     loss.backward()
        #     cur_loss = loss.item()
        #     optimizer.step()
        #
        #     st = ed
        #     batch_num += 1
        #     if ed < length and ed + bs >= length:
        #         ed += length - ed
        #     else:
        #         ed += bs

        # if (epoch + 1) % opt.args.upd == 0:
        #
        #     model_cpu = model.cpu()
        #     model_cpu.device = "cpu"
        #     # TODO Combine
        #     # model_cpu.a = model_cpu.a.cpu()
        #     # model_cpu.b = model_cpu.b.cpu()
        #     data = data.cpu()
        #     adj = adj.cpu()
        #
        #     model.eval()
        #     _, _, z, _, _ = model_cpu(data, adj)
        #     hidden_emb = z.cpu().data.numpy()
        #     # upth = update_threshold(upth, up_eta)
        #     # pos_inds = update_similarity(hidden_emb, upth, pos_num, A)
        #     upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
        #     pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num, A=None)
        #     pos_inds_device = torch.LongTensor(pos_inds).to(device)
        #
        #     print("Epoch: {}, train_loss={:.5f}".format(
        #         epoch, cur_loss))
        #
        #     kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
        #     res = kmeans.labels_
        #     acc, nmi, ari, f1 = eva(label, n_outliers, res, str(epoch))
        #     if acc != -1:
        #         #    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
        #         acc_result.append(acc)
        #         nmi_result.append(nmi)
        #         ari_result.append(ari)
        #         f1_result.append(f1)

        # TODO
        # loss：非正负样本生成
        if epoch % 20 == 0:

            model_cpu = model.cpu()
            model_cpu.device = "cpu"
            # TODO Combine
            # model_cpu.a = model_cpu.a.cpu()
            # model_cpu.b = model_cpu.b.cpu()
            data = data.cpu()
            adj = adj.cpu()

            model.eval()
            # _, _, z, _, q = model_cpu(data, adj)
            _, _, _, _, z, _, q_z = model(data, adj)

            # TODO
            # 聚类1
            # kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            # res = kmeans.labels_
            # 聚类2
            res = q_z.data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(label, n_outliers, res, str(epoch))
            if acc != -1:
                #    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
                acc_result.append(acc)
                nmi_result.append(nmi)
                ari_result.append(ari)
                f1_result.append(f1)
            # eva(y, res3, str(epoch) + 'P')

        model.train()

        # split batch
        index = list(range(len(data)))
        split_list = data_split(index, 2000)

        data = data.to(device)
        model = model.to(device)
        model.device = "cuda"
        # TODO Combine
        # model.a = model.a.to(device)
        # model.b = model.b.to(device)

        print("共有{}个batch".format(len(split_list)))

        batch_count = 0
        # mini-batch
        for batch in split_list:
            batch_count = batch_count + 1
            data_batch = data[batch]
            adj_batch = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj.to_dense()[batch, :][:, batch])).to(device)


            # 基础模型
            # GCN
            # z_hat, adj_hat, z, z_cluster, q = model(data_batch, adj_batch)
            # GAT
            # z_hat, adj_hat, z, _, _, _ = model(data_batch, adj_batch)
            # 组合模型3
            # h_hat, adj_hat_ae, z_gcn_hat, adj_hat_gcn, z_hat, adj_hat, z_ae, z_gcn, z, q_h, q_z = model(data_batch, adj_batch)
            # h_hat, adj_hat_ae, z_gat_hat, adj_hat_gat, _, adj_hat, z_ae, z_gat, z, q_h, q_z = model(data_batch, adj_batch)
            # 组合模型4/5
            h_hat, adj_hat_ae, z_hat, adj_hat, z, q_h, q_z = model(data_batch, adj_batch)
            # Combine2
            # h_hat, adj_hat_ae, z_gcn_hat, adj_hat_gcn, z_hat, adj_hat, z_ae, z_gcn, z, q_h, q_z, attention_alpha = model(data, adj)
            # Merge2 / Merge_ATT2
            # h_hat, adj_hat_ae, z_hat, adj_hat, z, q_h, q_z, attention_alpha = model(data, adj)

            # 基础loss
            # ----LossB_X
            # loss = F.mse_loss(z_hat, data)
            # print("loss-type: LossB_X")
            # ----LossB_A
            # loss = F.binary_cross_entropy(adj_hat.view(-1), adj_label.to_dense().view(-1))
            # print("loss-type: LossB_A")
            # ----LossB_AX
            # loss = F.mse_loss(z_hat, torch.spmm(adj, data))
            # print("loss-type: LossB_AX")
            # ----LossC_H_PQ / LossC_Z_PQ
            # loss = F.kl_div(q.log(), p, reduction='batchmean')
            # print("loss-type: LossC_H_PQ/LossC_Z_PQ")
            # ----LossC_H_PH / LossC_Z_PZ
            # loss = F.kl_div(z_cluster.log(), p, reduction='batchmean')
            # print("loss-type: LossC_H_PH/LossC_Z_PZ")
            # ----LossC'(H/Z -> P)
            # p = target_distribution_from_emb(z_cluster)
            # loss = F.kl_div(q.log(), p, reduction='batchmean')
            # print("loss-type: LossC'(H/Z -> P)")

            # TODO 修改loss 需要查看聚类方法
            # loss组合

            # loss_Z2X = F.mse_loss(z_hat, data_batch)
            # loss_H2X = F.mse_loss(h_hat, data_batch)
            # loss_Z2A = F.binary_cross_entropy(adj_hat.view(-1), adj_label.to_dense()[batch, :][:, batch].view(-1))
            # loss_Z2AX = F.mse_loss(z_hat, torch.spmm(adj_batch, data_batch))

            # ----- Effectiveness of loss functions.(paper)------
            # loss_H_A = loss_H2X + loss_Z2A
            # loss_Z_A = loss_Z2X + loss_Z2A
            # print("loss-type: loss_H/Z_A")
            # loss_H_AX = loss_H2X + loss_Z2AX
            # loss_Z_AX = loss_Z2X + loss_Z2AX
            # print("loss-type: loss_H/Z_AX")
            # loss_A_AX = loss_Z2A + loss_Z2AX
            # print("loss-type: loss_A_AX")
            # loss_H_A_AX =  loss_H2X + loss_Z2A + loss_Z2AX
            # loss_Z_A_AX = loss_Z2X + loss_Z2A + loss_Z2AX
            # print("loss-type: loss_H/Z_A_AX")

            p = target_distribution(q_z.data)
            loss_Z_PQ = F.kl_div(q_z.log(), p, reduction='batchmean')

            loss = loss_Z_PQ
            print("loss={}, loss-type: Loss_PQ(Z/H)".format(loss))

            # TODO 组合3
            # loss_Zgcn2X = F.mse_loss(z_gcn_hat, data)
            # loss_Zgcn2A = F.binary_cross_entropy(adj_hat_gcn.view(-1), adj_label.to_dense().view(-1))
            # loss_Zgcn2AX = F.mse_loss(z_gcn_hat, torch.spmm(adj, data))

            # loss_H_PQ = F.kl_div(q_h.log(), p_h, reduction='batchmean')
            # loss_Z_PQ = F.kl_div(q_z.log(), p_z, reduction='batchmean')
            # # ----Loss2 组合3/4/5
            # loss = loss_Z2X
            # print("组合3/4/5-loss-type: Loss2(Z→X)")
            # # ----Loss3 组合3
            # loss = loss_H2X + loss_Zgcn2A + loss_H_PQ + loss_Z_PQ
            # print("组合3-loss-type: Loss3(H→X，Zgcn→A, KL)")
            # # ----Loss3 组合4/5
            # loss = loss_H2X + loss_Z2A
            # loss = loss_Z2X + loss_Z2A
            # print("组合4/5-loss-type: Loss3(H/Z→X，Z→A)")
            # # ----Loss4 组合3
            # loss = loss_H2X + loss_Zgcn2AX + loss_H_PQ + loss_Z_PQ
            # print("组合3-loss-type: Loss4(H→X，Zgcn→AX，KL)")
            # # ----Loss4 组合4/5
            # loss = loss_H2X + loss_Z2AX
            # print("组合4/5-loss-type: Loss4(H→X，Z→AX)")
            # # ----Loss5 组合3
            # loss = loss_H2X + loss_Zgcn2X + loss_H_PQ + loss_Z_PQ
            # print("组合3-loss-type: Loss5(H→X，Zgcn→X，KL)")
            # # ----Loss5 组合4/5
            # loss = loss_H2X + loss_H_PQ + loss_Z_PQ
            # print("组合4/5-loss-type: Loss5(H→X，KL)")
            # # ----Loss6 组合4/5
            # loss = loss_Z2X + loss_H_PQ + loss_Z_PQ
            # print("组合4/5-loss-type: Loss6(Z→X，KL)")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    # return np.mean(acc_result), np.mean(nmi_result), np.mean(ari_result), np.mean(f1_result), len(acc_result), pos_inds, n, attention_alpha
    # return np.mean(acc_result), np.mean(nmi_result), np.mean(ari_result), np.mean(f1_result), len(acc_result), attention_alpha
    return np.mean(acc_result), np.mean(nmi_result), np.mean(ari_result), np.mean(f1_result), len(acc_result)