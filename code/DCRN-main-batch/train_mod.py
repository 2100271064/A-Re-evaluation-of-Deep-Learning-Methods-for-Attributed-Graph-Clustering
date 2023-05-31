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

        # split batch
        index = list(range(len(X_tilde1)))
        split_list = data_split(index, 2000)

        X_tilde1 = X_tilde1.to(opt.args.device)
        X_tilde2 = X_tilde2.to(opt.args.device)
        model = model.to(opt.args.device)

        print("共有{}个batch".format(len(split_list)))

        batch_count = 0
        # mini-batch
        for batch in split_list:

            batch_count = batch_count + 1

            X_tilde1_batch = X_tilde1[batch]
            X_tilde2_batch = X_tilde2[batch]

            Ad_batch = Ad[batch, :][:, batch]
            Ad_batch = Ad_batch.to(opt.args.device)

            Am_batch = Am[batch, :][:, batch]
            Am_batch = Am_batch.to(opt.args.device)

            # input & output
            X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1_batch, Ad_batch,
                                                                                     X_tilde2_batch, Am_batch)

            # calculate loss: L_{DICR}, L_{REC} and L_{KL}
            L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all)
            L_REC = reconstruction_loss(X[batch], A_norm.to_dense()[batch, :][:, batch], X_hat, Z_hat, A_hat)
            L_KL = distribution_loss(Q, target_distribution(Q[0].data))
            loss = L_DICR + L_REC + opt.args.lambda_value * L_KL

            print("当前epoch={}, 当前batch={}, loss={}".format(epoch, batch_count, loss))

            # optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if epoch % 20 == 0:
            model_cpu = model.cpu()
            X_tilde1 = X_tilde1.cpu()
            X_tilde2 = X_tilde2.cpu()
            Ad = Ad.cpu()
            Am = Am.cpu()

            with torch.no_grad():
                X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model_cpu(X_tilde1, Ad, X_tilde2, Am)

            acc, nmi, ari, f1, _ = clustering(Z, y, n_outliers, n_clusters)
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)

            Ad = Ad.to(opt.args.device)
            Am = Am.to(opt.args.device)

    print("Optimization Finished!")
    acc_arr = np.array(acc_list)
    nmi_arr = np.array(nmi_list)
    ari_arr = np.array(ari_list)
    f1_arr = np.array(f1_list)

    return np.mean(acc_arr), np.mean(nmi_arr), np.mean(ari_arr), np.mean(f1_arr)
