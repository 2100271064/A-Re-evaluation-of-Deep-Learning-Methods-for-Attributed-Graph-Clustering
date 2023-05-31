'''
Author: xinying Lai
Date: 2022-09-07 16:22:49
LastEditTime: 2022-09-13 10:23:17
Description: Do not edit
'''
import argparse

parser = argparse.ArgumentParser(description='DFCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# TODO
parser.add_argument('--name', type=str, default="pubmed")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--freedom_degree', type=float, default=1.0)
# TODO 原来是200
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--gamma_value', type=float, default=0.1)
parser.add_argument('--lambda_value', type=int, default=10)

parser.add_argument('--lr_usps', type=float, default=1e-3)
parser.add_argument('--lr_hhar', type=float, default=1e-3)
parser.add_argument('--lr_reut', type=float, default=1e-4)
parser.add_argument('--lr_acm', type=float, default=5e-5)
parser.add_argument('--lr_dblp', type=float, default=1e-4)
parser.add_argument('--lr_cite', type=float, default=1e-4)
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)
# TODO
# parser.add_argument('--version', type=str, default="max_cluster_num_5")
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--n_components', type=int, default=500)
args = parser.parse_args()