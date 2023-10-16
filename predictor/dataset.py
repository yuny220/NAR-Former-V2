import os
import copy
import onnx
import time
import torch
import random
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset as TorchDataset
from feature.graph_feature import extract_graph_feature
from feature.position_encoding import get_embedder
from torch.utils.data import Sampler

from data_process.data_and_encoding_generate import tokenizer, ac_aug_generate, padding_for_batch

def get_torch_data(onnx_file, batch_size, cost_time, embed_type):
    adjacent, node_features, static_features, topo_features = extract_graph_feature(onnx_file, batch_size, embed_type)

    edge_index = torch.from_numpy(np.array(np.where(adjacent > 0))).type(torch.long)

    node_features = np.array(node_features, dtype=np.float32) 
    node_features = torch.from_numpy(node_features).type(torch.float)
      
    x =  node_features

    sf = torch.from_numpy(static_features).type(torch.float)
    y = torch.FloatTensor([cost_time])
    data = Data(
        x = x,
        edge_index = edge_index,
        y = y,
    )
    return data, sf

class AccuracyDataset(TorchDataset):
    def __init__(self, dataset, part, data_path, percent=0, use_aug=False, dx=32, dp=32, embed_type=None, op_code_type='pe'):
        self.dx = dx
        self.dp = dp
        self.embed = embed_type
        self.optype = op_code_type
        self.part = part
        self.dataset = dataset
        self.data_path = data_path
        self.percent = percent
        self.use_aug = use_aug
        self.data = self.load()

    def load(self):
        datas = torch.load(self.data_path)

        loaded_data = []
        data_num = int(self.percent) if self.percent>1 else int(len(datas[0])*self.percent)
        if self.part == 'train':
            keys = list(range(data_num))
        elif self.part == 'val':
            keys = list(range(data_num, data_num + 200))
        elif self.part == 'test':
            keys = list(range(len(datas))) # test all
            #keys = list(range(data_num + 200, data_num + 300))
        
        total = 0
        for key in keys:
            example = copy.deepcopy(datas[key])
            example_tensor =self.preprocess(example)
            if self.use_aug and self.part=='train':
                auged_opss, auged_adjs = ac_aug_generate(datas[key]['ops'],\
                                        datas[key]['adj'], datas[key]['num_vertices'])
                if len(auged_adjs) > 0:
                    total+=1
                    auged_code = tokenizer(auged_opss[0], auged_adjs[0], self.dx, self.dp, self.embed, self.optype)
                    auged_code, auged_adj = padding_for_batch(auged_code, auged_adjs[0]) if self.dataset == 'nasbench101' \
                                            else (auged_code, auged_adjs[0])
                    
                    code_tensor = torch.tensor(auged_code, dtype=torch.float32)                       
                    adj_tensor = torch.tensor(auged_adj, dtype=torch.float32)
                    adj_tensor += torch.t(adj_tensor)
                    auged_tensor = (code_tensor, adj_tensor, example_tensor[2], example_tensor[3], example_tensor[4])
                else:
                    auged_tensor = example_tensor
                loaded_data.append([example_tensor, auged_tensor])
            else:
                loaded_data.append(example_tensor)
        print('Total auged data: ', total)
        return loaded_data
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self, data):
        if self.dataset == 'nasbench101':
            code, adj = padding_for_batch(data['netcode'], data['adj'])
            V_label = torch.tensor([data['validation_accuracy']], dtype=torch.float32)
            T_label = torch.tensor([data['test_accuracy']], dtype=torch.float32)
        elif self.dataset == 'nasbench201':
            code, adj = data['netcode'], data['adj']
            V_label= torch.tensor([data['validation_accuracy_avg']], dtype=torch.float32)
            T_label = torch.tensor([data['test_accuracy_avg']], dtype=torch.float32)


        code = torch.tensor(code, dtype=torch.float32)
        N = data['num_vertices']

        adj = torch.tensor(adj, dtype=torch.float32)
        adj += torch.t(adj)

        return code, adj, N, V_label, T_label

class GraphLatencyDataset(Dataset):
    # specific a platform
    def __init__(self, root, onnx_dir, latency_file, embed_type, selfpos, override_data=False, transform=None, pre_transform=None,
                model_types=None, train_test_stage=None, platforms=None, sample_num=-1):
        super(GraphLatencyDataset, self).__init__(root, transform, pre_transform)
        self.onnx_dir = onnx_dir
        self.latency_file = latency_file
        self.latency_ids = []
        self.override_data = override_data
        self.model_types = model_types
        self.train_test_stage = train_test_stage
        self.platforms = platforms

        self.embed_type = embed_type
        self.selfpos = selfpos

        # Load the gnn model for block
        self.device = None
        print("Extract input data from onnx...")
        self.custom_process()
        print("Done.")

        if isinstance(sample_num, int) and sample_num > 0:
            random.seed(1234)
            random.shuffle(self.latency_ids)
            self.latency_ids = self.latency_ids[:sample_num]
        
        # train_model_types == test_model_types
        if isinstance(sample_num, list):
            self.latency_ids = self.latency_ids[sample_num[0]:sample_num[1]]

        random.seed(1234)
        random.shuffle(self.latency_ids)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def custom_process(self):
        with open(self.latency_file) as f:
            for line in f.readlines():

                line = line.rstrip()
                items = line.split(" ")
                speed_id = str(items[0])
                graph_id = str(items[1])
                batch_size = int(items[2])
                cost_time = float(items[3])
                plt_id = int(items[5])

                if self.model_types and items[4] not in self.model_types:
                    continue

                if self.platforms and plt_id not in self.platforms:
                    continue

                if self.train_test_stage and items[6] != self.train_test_stage:
                    continue

                onnx_file = os.path.join(self.onnx_dir, graph_id)
                if os.path.exists(onnx_file):
                    data_file = os.path.join(self.processed_dir, '{}_{}_data.pt'.format(speed_id, plt_id))
                    sf_file = os.path.join(self.processed_dir, '{}_{}_sf.pt'.format(speed_id, plt_id))
                    graph_name = "{}_{}_{}".format(graph_id, batch_size, plt_id)
                    self.latency_ids.append((data_file, sf_file, graph_name, plt_id))

                    if (not self.override_data) and os.path.exists(data_file) and os.path.exists(sf_file):
                        continue

                    if len(self.latency_ids) % 1000 == 0:
                        print(len(self.latency_ids))

                    try:
                        print(onnx_file)
                        GG = onnx.load(onnx_file)
                        data, sf = get_torch_data(GG, batch_size, cost_time, self.embed_type)

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        torch.save(data, data_file)
                        torch.save(sf, sf_file)
                    except Exception as e:
                        self.latency_ids.pop()
                        print("Error", e)
        


    def len(self):
        return len(self.latency_ids)

    def get(self, idx):
        data_file, sf_file, graph_name, plt_id = self.latency_ids[idx]
        data = torch.load(data_file)
        sf = torch.load(sf_file)

        if self.selfpos > 0:
            fn, _ = get_embedder(self.selfpos, self.embed_type, input_type='np_array')
            sp_coding = [fn(np.array([int(i)], dtype='float32')) for i in range(data.x.size(0))]
            sp_coding = np.array(sp_coding, dtype=np.float32) 
            sp_tensor = torch.from_numpy(sp_coding).type(torch.float)
            data.x = torch.cat([data.x, sp_tensor], dim=-1)

        n_edges = data.num_edges
        return data, sf, n_edges, graph_name, plt_id


class FixedLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, include_partial=False, rng=None, maxlen=None,
                 length_to_size=None):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = { 0: self.batch_size }
        self.length_map = self.get_length_map()
        self.reset()
    def get_length_map(self):
        '''
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.
        '''
        # Record the lengths of each example.
        length_map = {} #{70:[0, 23, 3332, ...], 110:[3, 421, 555, ...], length:[dataidx_0, dataidx_1, ...]}
        for i in range(len(self.data_source)):
            length = self.data_source[i][2]
            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue
            length_map.setdefault(length, []).append(i)
        return length_map

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """
        # Shuffle the order.
        for length in self.length_map.keys():
            self.rng.shuffle(self.length_map[length])

        # Initialize state.
        state = {}
        for length, arr in self.length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = len(arr) % batch_size
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus'] >= torch.cuda.device_count():
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = self.length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1
        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)