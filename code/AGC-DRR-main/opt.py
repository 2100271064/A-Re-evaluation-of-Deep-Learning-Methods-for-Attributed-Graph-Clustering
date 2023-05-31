'''
Author: xinying Lai
Date: 2022-09-07 15:00:15
LastEditTime: 2022-09-11 21:53:44
Description: Do not edit
'''
import argparse

parser = argparse.ArgumentParser(description='DFCN1', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# TODO  
parser.add_argument('--name', type=str, default='uk')
# TODO 原来是 N2的学习率 citeseer 1e-4    其他 5e-4
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--k', type=int, default=None)
# TODO(代码中有)
parser.add_argument('--n_clusters', type=int, default=4)
parser.add_argument('--n_z', type=int, default=10)
# TODO PCA后的特征维数 没有使用
parser.add_argument('--n_input', type=int, default=100)
# parser.add_argument('--gamma_value', type=float, default=1)
# TODO(代码中有)
parser.add_argument('--data_path', type=str, default='.txt')
# TODO(代码中有)
parser.add_argument('--label_path', type=str, default='.txt')
# ？
# parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
# TODO PCA主成分分析 主要组件数 原来是50
parser.add_argument('--n_components', type=int, default=500)
# parser.add_argument('--batch_size', type=int, default=1600)
# TODO 原来是200
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--gae_n_enc_1', type=int, default=1000)
parser.add_argument('--gae_n_enc_2', type=int, default=500)
parser.add_argument('--gae_n_enc_3', type=int, default=500)
parser.add_argument('--emb_dim', type=int, default=500,help='embedding dimension')
# ？
# parser.add_argument('--dataset', type=str, default='ogbg-molesol',help='Dataset')
# TODO 原来是 N1的学习率 citeseer 1e-3     其他 1e-4
parser.add_argument('--view_lr', type=float, default=1e-4,help='View Learning rate.')
parser.add_argument('--num_gc_layers', type=int, default=5,help='Number of GNN layers before pooling')
parser.add_argument('--pooling_type', type=str, default='standard',help='GNN Pooling Type Standard/Layerwise')
parser.add_argument('--mlp_edge_model_dim', type=int, default=128,help='embedding dimension')
parser.add_argument('--pred_dim', type=int, default=64,help='embedding dimension')
parser.add_argument('--drop_ratio', type=float, default=0.0,help='Dropout Ratio / Probability')
parser.add_argument('--reg_lambda', type=float, default=1, help='View Learner Edge Perturb Regularization Strength')
parser.add_argument('--seed', type=int, default=0)
# 自己加的
# parser.add_argument('--version', type=str, default="max_cluster_num_5")
parser.add_argument('--gpu', type=str, default="3")
args = parser.parse_args()