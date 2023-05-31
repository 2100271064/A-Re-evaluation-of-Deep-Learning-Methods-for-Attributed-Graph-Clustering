'''
Author: xinying Lai
Date: 2022-09-07 15:00:17
LastEditTime: 2022-09-07 16:03:47
Description: Do not edit
'''
import torch
from torch.nn import Sequential, Linear, ReLU
from opt import  args
from torch import nn



class ViewLearner(torch.nn.Module):
	def __init__(self, encoder, mlp_edge_model_dim=64):
		super(ViewLearner, self).__init__()

		self.encoder = encoder
		self.input_dim = args.emb_dim

		self.mlp_edge_model = Sequential(
			# 输入：拼接后的边向量
			Linear(self.input_dim * 2, 1),
		#	ReLU(),
		#	Linear(mlp_edge_model_dim, 1)
		)
		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x,adj,edge_index):
		node_emb= self.encoder(x,adj)
		src, dst = edge_index[0], edge_index[1]
		emb_src = node_emb[src]
		emb_dst = node_emb[dst]
	#	print(emb_src.shape)
		# 边嵌入表示
		edge_emb = torch.cat([emb_src, emb_dst], 1)
	#	print(edge_emb.shape)
		# 边权重向量
		edge_logits = self.mlp_edge_model(edge_emb)

		return edge_logits