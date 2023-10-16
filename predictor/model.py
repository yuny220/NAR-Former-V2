import math
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter
from collections import defaultdict
from torch_geometric.nn.dense.linear import Linear
from typing import Optional

def gen_Khop_adj(edge_index, n_tokens, k=1):
    value = torch.ones(edge_index.size(1)).to(edge_index.device) # edge_index(2, num_edges)
    temp = torch.sparse_coo_tensor(edge_index, value, size=(n_tokens, n_tokens))
    matrix = temp.to_dense()
    
    if k == 1:
        return matrix


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return
    if init_type =='thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == 'kaiming_normal_in':
        nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_normal_out':
        nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_in':
        nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_out':
        nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')


class GNN_LinearAttn(torch.nn.Module):
    def __init__(self, in_dim, out_dim, normalize, degree=False, bias=True):
        super(GNN_LinearAttn, self).__init__()
        self.normalize = normalize
        self.degree = degree

        self.lin_l = Linear(in_dim, out_dim, bias=bias)

        self.lin_r = Linear(in_dim, out_dim, bias=False)

        if degree:
            self.lin_d = Linear(1, in_dim, bias=True)
            self.sigmoid_d = nn.Sigmoid()

        self.lin_qk = Linear(in_dim, in_dim, bias=True)
        self.sigmoid_qk = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.degree:
            self.lin_d.reset_parameters()

    def forward(self, x, A):
        # x (B, N, D)
        # A (B, N, N)
        if self.degree:
            degree = A.sum(dim=-1, keepdim=True) #(B, N, 1)
            degree = self.sigmoid_d(self.lin_d(degree))
            x = x * degree

        B, N, N = A.shape

        # Attn_V1 Aggregate
        QK = self.sigmoid_qk(self.lin_qk(x))
        scores = torch.matmul(QK, QK.transpose(-2, -1)) / math.sqrt(x.size(-1)) # (B, N, N)
        scores = scores * A

        attn = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.matmul(attn, x)
        out = self.lin_l(out)

        # Adj Feat + Root Feat
        out = out + self.lin_r(x)

        # L2 Normalization
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        
        # out (B, N, D2)
        return out


class GroupLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
        This is based on the Pyramidal recurrent unit paper:
            https://arxiv.org/abs/1808.09029
    '''

    def __init__(self, in_features: int, out_features: int, n_groups: int = 4, use_shuffle: bool = False,
                 norm_type: Optional[str] = None, use_bias: bool = False):
        '''

        :param in_features: number of input features
        :param out_features: number of output features
        :param n_groups: number of groups in GLT
        :param use_bias: use bias or not
        :param use_shuffle: shuffle features between different groups
        
        :param norm_type: Normalization type (e.g. LayerNorm)
        '''
        super(GroupLinear, self).__init__()

        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            raise Exception(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            raise Exception(err_msg)

        # warning_message = 'Please install custom cuda installation for faster training and inference'

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if use_bias:
            # add 1 in order to make it broadcastable across batch dimension
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        if norm_type is not None:
            if 'ln' in norm_type.lower():
                self.normalization_fn = nn.LayerNorm(out_groups)
                self.norm_type = norm_type
            else:
                raise NotImplementedError
        else:
            self.normalization_fn = None
            self.norm_type = None

        self.n_groups = n_groups
        self.use_bias = use_bias
        self.shuffle = use_shuffle
        self.feature_shuffle = True if use_shuffle else False

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        '''
        N --> Input dimension
        M --> Output dimension
        g --> groups
        G --> gates
        :param x: Input of dimension B x N
        :return: Output of dimension B x M
        '''
        bsz = x.size(0)
        # [B x N] --> [B x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B x g x N/g] --> [g x B  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B  x N/g] x [g x N/g x M/g] --> [g x B x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights

        # add bias
        if self.use_bias:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g x B x M/g] --> [B x M/g x g]
            x = x.permute(1, 2, 0)
            # [B x M/g x g] --> [B x g x M/g]
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            # [g x B x M/g] --> [B x g x M/g]
            x = x.transpose(0, 1)  # transpose so that batch is first

        # feature map normalization
        if self.normalization_fn is not None:
            x = self.normalization_fn(x)
        
        x = x.contiguous().view(bsz, -1)
        return x

    def forward(self, x):
        '''
        :param x: Input of shape [T x B x N] (should work with [B x T x N]
        :return:
        '''
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError

        return x
        

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GroupedFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, feat_shuffle, norm_type=None, num_groups=8, dropout = 0.):
        super().__init__()
        glt_up = GroupLinear(dim, hidden_dim, num_groups, feat_shuffle, norm_type, use_bias=True)
        glt_down = GroupLinear(hidden_dim, dim, num_groups, feat_shuffle, norm_type, use_bias=True)

        self.net = nn.Sequential(
            glt_up,
            nn.ReLU(),
            nn.Dropout(dropout),
            glt_down,
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# reduce_func: "sum", "mul", "mean", "min", "max"
class Net(torch.nn.Module):
    def __init__(
        self,
        dataset,
        feat_shuffle,
        glt_norm,
        n_attned_gnn=2,
        num_node_features=44,
        gnn_hidden=512,
        fc_hidden=512,
        use_degree=False,
        reduce_func="sum",
        norm_sf=False,
        ffn_ratio=4,
        init_values=1e-4,
        real_test=False,
    ):
        super(Net, self).__init__()
        self.dataset = dataset
        self.real_test = real_test
        self.reduce_func = reduce_func
        self.norm_sf = norm_sf
        self.n_attned_gnn = n_attned_gnn

        self.gnn_layers = nn.ModuleList()
        self.gnn_drops = nn.ModuleList()
        self.gnn_relus = nn.ModuleList()
        self.FFN_layers = nn.ModuleList()
        self.layer_scale = nn.ParameterList()

        for j in range(n_attned_gnn):
            gnn_dim_in = num_node_features if j==0 else gnn_hidden
            self.gnn_layers.append(
                GNN_LinearAttn(gnn_dim_in, gnn_hidden, True, use_degree))
            self.gnn_drops.append(nn.Dropout(p=0.05))
            self.gnn_relus.append(nn.ReLU())

            self.FFN_layers.append(PreNorm(gnn_hidden, GroupedFeedForward(gnn_hidden, gnn_hidden*ffn_ratio, feat_shuffle, glt_norm, dropout=0.05)))
            self.layer_scale.append(nn.Parameter(init_values * torch.ones((gnn_hidden)),requires_grad=True))            

        if self.dataset == 'nasbench101' or self.dataset == 'nasbench201':
            total_dim = n_attned_gnn * gnn_hidden
            self.dim_out = gnn_hidden
            self.fusion_mlp = nn.Sequential(
                                nn.Linear(total_dim, total_dim//4), 
                                nn.ReLU(),
                                nn.Linear(total_dim//4, n_attned_gnn),
                                nn.Sigmoid())

        out_dim = 1
        if self.norm_sf:
            self.norm_sf_linear = nn.Linear(40, gnn_hidden)
            self.norm_sf_drop = nn.Dropout(p=0.05)
            self.norm_sf_relu = nn.ReLU()
            sf_hidden = gnn_hidden
        else:
            sf_hidden = 4 if dataset=='nnlqp' else 0
        self.fc_1 = nn.Linear(gnn_hidden + sf_hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.predictor = nn.Linear(fc_hidden, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas", "relu")
                init_tensor(m.bias, "thomas", "relu")

    def forward(self, data1, data2, n_edges):
        gnn_feat = []

        if self.dataset == 'nnlqp':
            data, static_feature = data1, data2
            
            # feat (B*N, D); N=21, 124, ...
            # adj (2, B*n_edges)
            # n_edges (B): (n_edges1, n_edges2, ....)
            feat, adj = data.x, data.edge_index
            
            idx = torch.ones(data.batch.size(0)).to(feat.device)
            idx = scatter(idx, data.batch, dim=0, reduce='sum') # (N1, N2, ...Ni...) length of each sample in batch

            for i in range(idx.size(0)):
                if i == idx.size(0)-1:
                    n_nodes_i = int(sum(idx[:i]))
                    n_edges_i = int(sum(n_edges[:i]))
                    x_i = feat[n_nodes_i : ].unsqueeze(0)
                    adj_i = gen_Khop_adj( adj[:, n_edges_i : ] - n_nodes_i, int(idx[i])).unsqueeze(0)
                else:
                    n_nodes_i, n_nodes_ii = int(sum(idx[:i])), int(sum(idx[:i+1]))
                    n_edges_i, n_edges_ii = int(sum(n_edges[:i])), int(sum(n_edges[:i+1]))
                    x_i = feat[n_nodes_i : n_nodes_ii].unsqueeze(0) # x_i (1, Ni, D)
                    adj_i = gen_Khop_adj( adj[:, n_edges_i : n_edges_ii ] - n_nodes_i, int(idx[i])).unsqueeze(0)

                x = x_i

                for gnn, dropout, relu, ffn, gamma in zip(self.gnn_layers, self.gnn_drops,
                                            self.gnn_relus, self.FFN_layers, self.layer_scale):
                    x = gnn(x, adj_i)
                    x = relu(x)
                    x = dropout(x) # x (1, Ni, D2)
                    x_ = ffn(x)
                    x = x + gamma * x_
                
                if self.reduce_func == 'sum':
                    x = x.sum(dim=1, keepdim=False) # x (1, D)
                else:
                    raise NotImplementedError
                
                gnn_feat.append(x)
            
            gnn_feat = torch.cat(gnn_feat, dim=0) # x(B, D)

            if self.norm_sf:
                static_feature = self.norm_sf_linear(static_feature)
                static_feature = self.norm_sf_drop(static_feature)
                static_feature = self.norm_sf_relu(static_feature)
            x = torch.cat([gnn_feat, static_feature], dim=1)
        
        elif self.dataset == 'nasbench101' or self.dataset == 'nasbench201':
            netcode, adj = data1, data2
            n_samples = netcode.size(0) if not self.real_test else 1
            for i in range(n_samples):
                layer_feats = []
                if self.real_test:
                    # For testing stage
                    adj_i = adj[:, :n_edges[i], :n_edges[i]]
                    x = netcode[:, :n_edges[i], :]
                else:
                    adj_i = adj[i, :n_edges[i], :n_edges[i]].unsqueeze(0)
                    x = netcode[i, :n_edges[i], :].unsqueeze(0)
                
                for gnn, dropout, relu, ffn, gamma in zip(self.gnn_layers, self.gnn_drops,
                                            self.gnn_relus, self.FFN_layers, self.layer_scale):
                    x = gnn(x, adj_i)
                    x = relu(x)
                    x = dropout(x) # x (1, Ni, D2)
                    x_ = ffn(x)
                    x = x + gamma * x_
                    layer_feats.append(x.mean(dim=1, keepdim=False)) # [(1, D)]

                gnn_feat.append(torch.cat(layer_feats, dim=-1)) # (1, D*num_layers)
            
            gnn_feat = torch.cat(gnn_feat, dim=0) # x(B, D*num_layers)
            
            layer_weights = self.fusion_mlp(gnn_feat) # (B, num_layers)

            # (B, num_layers, D) * (B, num_layers, 1) --> sum(1, False) --> (B, D)
            x = (gnn_feat.contiguous().view(-1, self.n_attned_gnn, self.dim_out) * layer_weights.unsqueeze(2)).sum(dim=1, keepdim=False)

        else:
            raise NotImplementedError

        x = self.fc_1(x)
        x = self.fc_relu1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu2(x)
        feat = self.fc_drop_2(x)
        x = self.predictor(feat)

        pred = -F.logsigmoid(x)
        return pred


class SRLoss(nn.Module):
    def __init__(self):
        super(SRLoss, self).__init__()
        self.cal_loss = nn.L1Loss()

    def forward(self, predicts, target):
        B = predicts.shape[0]
        ori_pre = predicts
        ori_tar = target
        index = list(range(B))
        random.shuffle(index)
        predicts = predicts[index]
        target = target[index]
        v1 = ori_pre - predicts
        v2 = ori_tar - target
        loss = self.cal_loss(v1, v2)
        return loss