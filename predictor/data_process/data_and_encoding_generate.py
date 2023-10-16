from ast import arg
from operator import ne
import os
import numpy as np
import torch
import random
import argparse
import sys
import math 
import itertools

MAX_LEN = 7

def transform_operations_category_101(ops):
    transform_dict =  {'input':0, 'conv1x1-bn-relu':1, 'conv3x3-bn-relu':2, 'maxpool3x3':3, 'output':4}
    res_ops = []
    res_ops = [transform_dict[k] for k in ops]
    return res_ops

def transform_operations_category_201(ops):
    transform_dict =  {'input':0, 'nor_conv_1x1':1, 'nor_conv_3x3':2, 'avg_pool_3x3':3, 'skip_connect':4, 'none':5, 'output':6}
    res_ops = []
    res_ops = [transform_dict[k] for k in ops]
    return res_ops

def padding_for_batch(code, adj):
    if len(adj) < MAX_LEN:
        for i in range(MAX_LEN - len(adj)):
            for l in adj:
                l.append(0)
        adj.extend([[0]*MAX_LEN for _ in range(MAX_LEN - len(adj))])

        code_ = np.zeros((MAX_LEN, code.shape[1]))
        code_[:code.shape[0], :] = code
        return code_, adj
    else:
        return code, adj
        
def tokenizer(ops, adj, dx, dp, embed_type='nerf', op_code_type='pe'):
    n_nodes = len(ops)

    # Index list to one-hot
    if op_code_type == 'onehot':
        ops_coding = np.eye(dx, dtype="float32")[ops]
        dim_x = dx
    # Index list to embedding, dim = 2*dx
    else:
        fn, _ = get_embedder(dx, embed_type=embed_type)
        code_ops_tmp = []
        for op in ops:
            code_ops_tmp.append(fn(np.array([int(op)], dtype="float32")))
        ops_coding = np.stack(code_ops_tmp, axis=0)
        dim_x = dx*2
    coding = ops_coding

    if dp > 0:
        fn, _ = get_embedder(dp, embed_type=embed_type)
        code_pos_tmp = []
        for i in range(n_nodes):
            code_pos_tmp.append(fn(np.array([int(i)], dtype="float32")))
        code_pos = np.stack(code_pos_tmp, axis=0)
        coding = np.concatenate([ops_coding, code_pos], axis=-1)
    dim_p = dp*2

    return coding


def get_embedder(multires, embed_type='nerf', input_type='numpy', i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'input_type' : input_type,
                'embedding_type' : embed_type,
                'include_input' : False,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : False,
    }
    if input_type=='tensor': 
        embed_kwargs['periodic_fns'] = [torch.sin, torch.cos] 
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed_tensor(x)
    else:
        embed_kwargs['periodic_fns'] = [np.sin, np.cos]
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
    
    return embed, embedder_obj.out_dim

class Embedder():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims'] #d=3
        out_dim = 0
        if self.kwargs['include_input']: #True
            embed_fns.append(lambda x : x)
            out_dim += d #out_dim=3
            
        max_freq = self.kwargs['max_freq_log2'] #max_freq=multires-1=9
        N_freqs = self.kwargs['num_freqs'] #N_freqs=multires=10
        
        dty = self.kwargs['input_type']
        if self.kwargs['embedding_type'] == 'nerf':
            if self.kwargs['log_sampling']: #True
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs) if dty=='tensor'\
                            else  2.**np.linspace(0., max_freq, num=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs) if dty=='tensor'\
                             else np.linspace(2.**0., 2.**max_freq, num=N_freqs)
            
            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']: #p_fn=torch.sin, p_fn=torch.cos
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * math.pi * freq))
                    out_dim += d 
        
        elif self.kwargs['embedding_type'] == 'trans':
            dim = self.kwargs['num_freqs']
            freq_bands = [ 1 / (10000**(j/dim)) for j in range(dim)]
            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']: #p_fn=torch.sin, p_fn=torch.cos
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d 

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed_tensor(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    def embed(self, inputs):
        return np.concatenate([fn(inputs) for fn in self.embed_fns])


def upper_tri_matrix(matrix):
    flag = True
    for i in range(len(matrix)):
        for j in range(0,i):
            if matrix[i][j] != 0:
                flag = False
                break
    return flag

def ac_aug_generate(ops, adj, num_vertices):
    temp = [i for i in range(1, num_vertices-1)]
    temp_list = itertools.permutations(temp)
    auged_adjs = [adj]
    auged_opss = [ops]

    for id, label in enumerate(temp_list):
        if len(auged_adjs)==2:
            break
        label = [0] + list(label) + [num_vertices-1]
        P = np.zeros((num_vertices, num_vertices))
        for i,j in enumerate(label):
            P[i, j] = 1
        P_inv = np.linalg.inv(P)
        adj_aug = (P@adj@P_inv).astype(int).tolist()
        ops_aug =(ops@P_inv).astype(int).tolist()
        if ((adj_aug not in auged_adjs) or (ops_aug not in auged_opss)) and upper_tri_matrix(adj_aug):
            auged_adjs.append(adj_aug)
            auged_opss.append(ops_aug)
    return auged_opss[1:], auged_adjs[1:]

def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nasbench101', help='dataset type')
    parser.add_argument('--data_path', type=str, default='nasbench101.json', help='path of json file')
    parser.add_argument('--save_dir', type=str, default='.', help='path of generated pt files')
    parser.add_argument('--multires_x', type=int, default=5, help='dim of operation encoding')
    parser.add_argument('--multires_p', type=int, default=0, help='dim of self position encoding')
    parser.add_argument('--embed_type', type=str, default=None, help='Type of position embedding: nerf|trans')
    parser.add_argument('--op_code_type', type=str, default='onehot', help='onehot|pe')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import json
    r = random.random
    random.seed(2022)
    args = argLoader()

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dx, dp = args.multires_x, args.multires_p

    with open(args.data_path) as f:
        archs = json.load(f)
    num = len(archs)
    print('Total samples: %d' % num)
    data = {}
    for i in range(len(archs)):
        print('Loading %d/%d' % (i, num))
        if args.dataset == 'nasbench101':
            ops = archs[str(i)]['module_operations']
            ops = transform_operations_category_101(ops)
            adj = archs[str(i)]['module_adjacency']
            data[i] = {
            'index': i,
            'adj': adj,
            'ops': ops,
            'num_vertices': len(ops),
            'params': archs[str(i)]['parameters'],
            'validation_accuracy': archs[str(i)]['validation_accuracy'],
            'test_accuracy': archs[str(i)]['test_accuracy'],
            'training_time': archs[str(i)]['training_time'],
            'netcode': tokenizer(ops, adj, dx, dp, args.embed_type, args.op_code_type)
                        }
        elif args.dataset == 'nasbench201':
            ops = archs[str(i)]['module_operations']
            ops = transform_operations_category_201(ops)
            adj = archs[str(i)]['module_adjacency']
            data[i] = {
                'index': i,
                'adj': adj,
                'ops': ops,
                'num_vertices': len(ops),
                'training_time': archs[str(i)]['training_time'],
                'test_accuracy': archs[str(i)]['test_accuracy'],
                'test_accuracy_avg': archs[str(i)]['test_accuracy_avg'],
                'validation_accuracy': archs[str(i)]['validation_accuracy'],
                'validation_accuracy_avg': archs[str(i)]['validation_accuracy_avg'],
                'netcode' : tokenizer(ops, adj, dx, dp, args.embed_type, args.op_code_type),

                'validation_loss': archs[str(i)]['valid_loss'],
                'test_loss': archs[str(i)]['test_loss']
                }
    torch.save(data, os.path.join(save_dir, f'all_{args.dataset}_optype({args.op_code_type}_{dx})_sp(2*{dp})_{args.embed_type}.pt'))
    