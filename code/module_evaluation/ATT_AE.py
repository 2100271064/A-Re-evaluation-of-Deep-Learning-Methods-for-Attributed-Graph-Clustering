import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 自己的学习权重
        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        # 邻居的学习权重
        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        # 网络的嵌入表示
        h = torch.mm(input, self.W)
        #前馈神经网络
        # 自己表示向量*重要因子
        attn_for_self = torch.mm(h,self.a_self)       #(N,1)
        # 邻居表示向量*重要因子
        attn_for_neighs = torch.mm(h,self.a_neighs)   #(N,1)
        # 拼接
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)
        # 强化结构
        if M != None:
            attn_dense = torch.mul(attn_dense,M)
        # 激活函数
        attn_dense = self.leakyrelu(attn_dense)            #(N,N)

        #--------掩码（邻接矩阵掩码）
        # 全1矩阵(*-9e15)，shape=adj.shape
        zero_vec = -9e15*torch.ones_like(adj)
        # adj为true的在zero_vec的对应位置 填入attn_dense对应位置的值
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        # 归一化
        attention = F.softmax(adj, dim=1)
        # *注意力权重
        h_prime = torch.matmul(attention,h)

        if concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
            return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

class ATT_AE(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(ATT_AE, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden_size, embedding_size, alpha)
        self.enc_cluster = GraphAttentionLayer(embedding_size, num_clusters, alpha)
        # 解码
        self.conv3 = GraphAttentionLayer(embedding_size, hidden_size, alpha)
        self.conv4 = GraphAttentionLayer(hidden_size, num_features, alpha)

        self.v = v
        self.num_clusters = num_clusters
        # cluster layer shape=(n_clusters, embedding_size)
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M=None):
        h, _ = self.conv1(x, adj, M)
        h, attention_alpha = self.conv2(h, adj, M)
        h_cluster, _ = self.enc_cluster(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        z_cluster = F.normalize(h_cluster, p=2, dim=1)
        A_hat = dot_product_decode(z)

        # 广播机制:
        # q.shape (N, 1, embedding_size)-(n_clusters, embedding_size)=(N, n_clusters, embedding_size)
        # q.shape (N, n_clusters, embedding_size)->(N, n_clusters)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        # q.shape (N, n_clusters)
        q = (q.t() / torch.sum(q, 1)).t()

        # 解码
        z_hat, _ = self.conv3(h, adj, M)
        z_hat, _ = self.conv4(z_hat, adj, M)

        return z_hat, A_hat, z, z_cluster, q, attention_alpha