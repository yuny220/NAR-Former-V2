import json
from collections import OrderedDict
import numpy as np
from nas_201_api import NASBench201API as API

api = API('/home/disk/NasBench201/NAS-Bench-201-v1_1-096897.pth', verbose=False)

'''
num = len(api)
for i, arch_str in enumerate(api):
  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))

# show all information for a specific architecture
api.show(1)
api.show(2)

# show the mean loss and accuracy of an architecture
#info = api.query_meta_info_by_index(1)  # This is an instance of `ArchResults`
#res_metrics = info.get_metrics('cifar10', 'train') # This is a dict with metric names as keys
#cost_metrics = info.get_comput_costs('cifar100') # This is a dict with metric names as keys, e.g., flops, params, latency

# get the detailed information
results = api.query_by_index(1, 'cifar100') # a dict of all trials for 1st net on cifar100, where the key is the seed
print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
for seed, result in results.items():
  print ('Latency : {:}'.format(result.get_latency()))
  print ('Train Info : {:}'.format(result.get_train()))
  print ('Valid Info : {:}'.format(result.get_eval('x-valid')))
  print ('Test  Info : {:}'.format(result.get_eval('x-test')))
  # for the metric after a specific epoch
  print ('Train Info [10-th epoch] : {:}'.format(result.get_train(10)))

print('='*100)

'''
more_info = api.get_more_info(1, 'cifar10-valid', None, '200', False)
for key in more_info.keys():
  print(key, ' : ', more_info[key])

def train_and_eval(arch_index, nepoch=None, dataname=None, use_converged_LR=True):
    assert dataname !='cifar10', 'Do not allow cifar10 dataset'
    if use_converged_LR and dataname=='cifar10-valid':
        assert nepoch == None, 'When using use_converged_LR=True, please set nepoch=None, use 12-converged-epoch by default.'
        print(arch_index, dataname)
        info = api.get_more_info(arch_index, dataname, None, '12', True)
        valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
        valid_acc_avg = api.get_more_info(arch_index, 'cifar10-valid', None, '200', False)['valid-accuracy']
        test_acc = api.get_more_info(arch_index, 'cifar10', None, '200', True)['test-accuracy']
        test_acc_avg = api.get_more_info(arch_index, 'cifar10', None, '200', False)['test-accuracy']
        valid_loss_avg = api.get_more_info(arch_index, 'cifar10-valid', None, '200', False)['valid-loss']
        test_loss_avg = api.get_more_info(arch_index, 'cifar10-valid', None, '200', False)['test-loss']

    else:
      raise ValueError('NOT IMPLEMENT YET')
    return valid_acc, valid_acc_avg, time_cost, test_acc, test_acc_avg, valid_loss_avg, test_loss_avg

def info2mat(idx):
    adj_mat = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1 ,0 ,0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0]])

    info = api.query_meta_info_by_index(idx)

    nodes = ['input']
    steps = info.arch_str.split('+')
    steps_coding = ['0', '0', '1', '0', '1', '2']
    cont = 0
    for step in steps:
        step = step.strip('|').split('|')
        for node in step:
            n, idx = node.split('~') #n: operation, idx: previous node
            assert idx == steps_coding[cont]
            cont += 1
            nodes.append(n)
    nodes.append('output')  # nodes: operation strings

    valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg, valid_loss_avg, test_loss_avg = train_and_eval(idx, nepoch=None, dataname='cifar10-valid', use_converged_LR=True)
    informations = { 'test_accuracy': test_acc * 0.01,
                      'test_accuracy_avg': test_acc_avg * 0.01,
                      'validation_accuracy':valid_acc * 0.01,
                      'validation_accuracy_avg': val_acc_avg * 0.01,
                      'module_adjacency':adj_mat.tolist(),
                      'module_operations': nodes,
                      #'module_operations': node_mat.tolist(),
                      #'module_operations': ops_idx,
                      'training_time':  time_cost,
                      'test_loss': valid_loss_avg,
                      'val_loss': test_loss_avg}

    return informations
      
def enumerate_dataset(dataset):
    for k in range(len(api)):
        print('{}: {}/{}'.format(dataset, k,len(api)))
        res = info2mat(k)
        yield {k:res}

def gen_json_file(dataset):
    data_dict = OrderedDict()
    enum_dataset = enumerate_dataset(dataset)
    for data_point in enum_dataset:
        data_dict.update(data_point)
    with open('{}.json'.format(dataset), 'w') as outfile:
        json.dump(data_dict, outfile)
    
if __name__=='__main__':
    gen_json_file('nasbench201_cifar10_valid_converged')