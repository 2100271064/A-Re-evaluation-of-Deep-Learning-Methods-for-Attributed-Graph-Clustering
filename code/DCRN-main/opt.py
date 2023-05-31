import argparse

parser = argparse.ArgumentParser(description='DCRN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting

# TODO
parser.add_argument('--name', type=str, default="dblp")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=3)
# TODO PUBMED: 0.1  other: 0.2(程序已经设定好)
parser.add_argument('--alpha_value', type=float, default=0.2)
parser.add_argument('--lambda_value', type=float, default=10)
parser.add_argument('--gamma_value', type=float, default=1e3)
# TODO DBLP: 1e-4    ACM: 5e-5   AMAP: 1e-3  CITE, PUBMED, CORAFULL: 1e-5(程序已经设定好)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--epoch', type=int, default=400)
# 原本是False
parser.add_argument('--show_training_details', type=bool, default=True)


# AE structure parameter from DFCN
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

# parser.add_argument('--version', type=str, default="max_cluster_num_5")
parser.add_argument('--gpu', type=str, default="0,1")
parser.add_argument('--n_components', type=int, default=500)
args = parser.parse_args()
