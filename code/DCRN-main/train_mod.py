'''
Author: xinying Lai
Date: 2022-09-13 14:10:27
LastEditTime: 2022-09-13 15:32:26
Description: Do not edit
'''
import tqdm
from utils_mod import *
from torch.optim import Adam


def train(model, X, y, A, A_norm, Ad, n_clusters, n_outliers):
    """
    train our model
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrix
        y: input label
        A: input origin adj
        A_norm: normalized adj
        Ad: graph diffusion
    Returns: acc, nmi, ari, f1
    """
    print("Training…")
    # calculate embedding similarity and cluster centers
    sim, centers = model_init(model, X, y, A_norm, n_clusters, n_outliers)

    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)

    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    Am = remove_edge(A, sim, remove_rate=0.1)

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    for epoch in tqdm.tqdm(range(opt.args.epoch)):
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        # input & output
        X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1, Ad, X_tilde2, Am)

        # calculate loss: L_{DICR}, L_{REC} and L_{KL}
        L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all)
        L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss = L_DICR + L_REC + opt.args.lambda_value * L_KL

        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clustering & evaluation
        acc, nmi, ari, f1, _ = clustering(Z, y, n_outliers, n_clusters)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)

        # if acc > opt.args.acc:
        #     opt.args.acc = acc
        #     opt.args.nmi = nmi
        #     opt.args.ari = ari
        #     opt.args.f1 = f1

    print("Optimization Finished!")
    acc_arr = np.array(acc_list)
    nmi_arr = np.array(nmi_list)
    ari_arr = np.array(ari_list)
    f1_arr = np.array(f1_list)

    return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr)
