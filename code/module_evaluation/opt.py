import argparse

parser = argparse.ArgumentParser(description='Laixy', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting

# TODO
parser.add_argument('--name', type=str, default="acm")

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--gpu', type=str, default="3")

parser.add_argument('--seed', type=int, default=3)
# TODO
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--upd', type=int, default=1, help='Update epoch.')

# 正样本
parser.add_argument('--upth_st', type=float, default=0.0011, help='Upper Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.0010, help='Upper Threshold end.')
# 负样本
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')

# ATT-AE structure parameter from DAEGC
parser.add_argument('--hidden1_dim', type=int, default=256)
# TODO
parser.add_argument('--hidden2_dim', type=int, default=16)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

# GCN structure parameter from DCRN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
# TODO
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# 组合dim提升
parser.add_argument('--n_enc_1', type=int, default=500)
parser.add_argument('--n_enc_2', type=int, default=500)
parser.add_argument('--n_enc_3', type=int, default=2000)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--n_dec_1', type=int, default=2000)
parser.add_argument('--n_dec_2', type=int, default=500)
parser.add_argument('--n_dec_3', type=int, default=500)

# AGE dim
# GCN/GAT 的embedding_dim = 500
# parser.add_argument('--age_dim', type=int, default=500)
# from AGE
# parser.add_argument('--bs', type=int, default=1000, help='Batchsize.')
# TODO(Train_AGE)
# parser.add_argument('--gae_n_enc_3', type=int, default=500)
# TODO(Train_AGE)
# parser.add_argument('--hidden2_dim', type=int, default=500)
parser.add_argument('--n_components', type=int, default=500)

args = parser.parse_args()
