import os
import time
import copy
import random
import torch
import logging
import argparse
import numpy as np
from scipy.stats import stats
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from model import SRLoss
from dataset import GraphLatencyDataset, AccuracyDataset, FixedLengthBatchSampler

from transformers import get_linear_schedule_with_warmup

class Metric(object):
    def __init__(self):
        self.all = self.init_pack()
        self.plts = {}

    def init_pack(self):
        return {
            'ps' : [],
            'gs' : [],
            'cnt': 0,
            'apes': [],                                # absolute percentage error
            'errbnd_cnt': np.array([0.0, 0.0, 0.0]),   # error bound count
            'errbnd_val': np.array([0.1, 0.05, 0.01]), # error bound value: 0.1, 0.05, 0.01
        }

    def update_pack(self, ps, gs, pack):
        for i in range(len(ps)):
            ape = np.abs(ps[i] - gs[i]) / gs[i]
            pack['errbnd_cnt'][ape <= pack['errbnd_val']] += 1
            pack['apes'].append(ape)
            pack['ps'].append(ps[i])
            pack['gs'].append(gs[i])
        pack['cnt'] += len(ps)

    def measure_pack(self, pack):
        acc = np.mean(pack['apes'])
        err = (pack['errbnd_cnt'] / pack['cnt'])
        tau = stats.kendalltau(pack['gs'], pack['ps']).correlation
        return acc, err, tau

    def update(self, ps, gs, plts=None):
        self.update_pack(ps, gs, self.all)
        if plts:
            for idx, plt in enumerate(plts):
                if plt not in self.plts:
                    self.plts[plt] = self.init_pack()
                self.update_pack([ps[idx]], [gs[idx]], self.plts[plt])

    def get(self, plt=None):
        if plt is None:
            return self.measure_pack(self.all)
        else:
            return self.measure_pack(self.plts[plt])


class Trainer(object):

    def __init__(self):
        self.args = self.init_args()
        self.logger = self.init_logger()        
        self.logger.info("Loading args: \n{}".format(self.args))

        #if self.args.gpu >= 0:
            #os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.args.gpu)
            #if not torch.cuda.is_available():
                #self.logger.error("No GPU={} found!".format(self.args.gpu))

        self.logger.info("Loading dataset:")
        if self.args.dataset == 'nnlqp':
            if self.args.train_test_stage:
                self.train_loader, self.test_loader = self.init_train_test_dataset()
            else:
                self.train_loader, self.test_loader = self.init_unseen_structure_dataset()
        else:
            self.init_accuracy_dataset()

        self.device = torch.device(f'cuda:{self.args.gpu}' if self.args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
        from model import Net
        self.model = Net(
            self.args.dataset,
            feat_shuffle=self.args.feat_shuffle,
            glt_norm=self.args.glt_norm,
            n_attned_gnn=self.args.n_attned_gnn,
            num_node_features=self.args.num_node_features,
            gnn_hidden=self.args.hidden_size,
            fc_hidden=self.args.hidden_size,
            use_degree=self.args.use_degree,
            norm_sf=self.args.norm_sf,
            ffn_ratio=self.args.ffn_ratio,
            real_test = self.args.only_test
        )
        self.model = self.model.to(self.device)
        self.logger.info("Init Model: {}".format(self.model))

        if not self.args.only_test:
            total_iters = self.args.epochs * len(self.train_loader)
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_rate*total_iters, num_training_steps=total_iters)
            if self.args.lambda_sr > 0:
                self.Loss_SR = SRLoss()
        self.logger.info("Loading model:")
        self.init_model()
        self.best_err = 0

    def init_args(self):        
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--gpu', type=int, default=0, help="gpu id, < 0 means no gpu")
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--warmup_rate', type=float, default=0.1)

        parser.add_argument('--norm_sf', action='store_true')
        parser.add_argument('--use_degree', action='store_true')

        parser.add_argument('--dataset', type=str, default='nnlqp', help='nnlqp|nasbench101|nasbench201')
        parser.add_argument('--data_root', type=str)
        parser.add_argument('--all_latency_file', type=str)
        parser.add_argument('--test_model_type', type=str)
        parser.add_argument('--train_model_types', type=str)
        parser.add_argument('--train_test_stage', action='store_true')
        parser.add_argument('--train_num', type=int, default=-1)

        parser.add_argument('--onnx_dir', type=str)
        parser.add_argument('--override_data', action='store_true')

        parser.add_argument('--log', type=str)
        parser.add_argument('--pretrain', type=str)
        parser.add_argument('--resume', type=str)
        parser.add_argument('--model_dir', type=str)

        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--ckpt_save_freq', type=int, default=20)
        parser.add_argument('--test_freq', type=int, default=10)
        parser.add_argument('--only_test', action='store_true')
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--num_node_features', type=int, default=44)

        # args for NAR-Former V2
        parser.add_argument('--embed_type', type=str)
        parser.add_argument('--n_attned_gnn', type=int, default=2)
        parser.add_argument('--feat_shuffle', type=bool, default=False)
        parser.add_argument('--ffn_ratio', type=int, default=4)
        parser.add_argument('--glt_norm', type=str, default=None)
        

        # args for NasBench 
        parser.add_argument('--multires_x', type=int, default=0)
        parser.add_argument('--multires_p', type=int, default=0)
        parser.add_argument('--optype', type=str, default='onehot')
        parser.add_argument('--lambda_sr', type=float, default=0, help='only be used in accuracy prediction')
        parser.add_argument('--lambda_cons', type=float, default=0, help='only be used in accuracy prediction')

        args = parser.parse_args()
        args.norm_sf = True if args.norm_sf else False
 
        return args


    def init_logger(self):
        if not os.path.exists("log"):
            os.makedirs("log")

        logger = logging.getLogger("FEID")
        logger.setLevel(level = logging.INFO)
        formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d" \
                                      "-%(levelname)s-%(message)s")

        # log file stream
        handler = logging.FileHandler(self.args.log)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        # log console stream
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
    
        logger.addHandler(handler)
        logger.addHandler(console)
    
        return logger

    def init_accuracy_dataset(self):
        if self.args.only_test:
            test_set = AccuracyDataset(self.args.dataset, "test", self.args.data_root, self.args.train_num)
            test_sampler = FixedLengthBatchSampler(test_set, self.args.batch_size, include_partial=True)
            self.test_loader = TorchDataLoader(test_set, shuffle=(test_sampler is None), batch_sampler=test_sampler)
            self.logger.info("Test data = {}".format(len(test_set)))

        else:
            train_set = AccuracyDataset(self.args.dataset, "train", self.args.data_root, self.args.train_num, self.args.lambda_cons>0, \
                                        self.args.multires_x, self.args.multires_p, self.args.embed_type, self.args.optype)
            test_set = AccuracyDataset(self.args.dataset, "val", self.args.data_root, self.args.train_num) # Valition dataset
            self.logger.info("Train data = {}, Val data = {}".format(len(train_set), len(test_set)))
            self.train_loader = TorchDataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
            self.test_loader = TorchDataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)

    def init_unseen_structure_dataset(self):
        model_types = set()
        for line in open(self.args.all_latency_file).readlines():
            model_types.add(line.split()[4])
        assert self.args.test_model_type in model_types
        test_model_types = set([self.args.test_model_type])
        if self.args.train_model_types:
            train_model_types = set(self.args.train_model_types.split(','))
            train_model_types = train_model_types & model_types
        else:
            train_model_types = model_types - test_model_types
        assert len(train_model_types) > 0

        self.logger.info("Train model types: {}".format(train_model_types))
        self.logger.info("Test model types: {}".format(test_model_types))

        sample_num_tr = [0,    1600] if train_model_types == test_model_types else -1
        sample_num_te = [1600, 2000] if train_model_types == test_model_types else -1

        train_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p, 
            override_data=self.args.override_data,
            model_types=train_model_types,
            sample_num=sample_num_tr,
        )
        test_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p, 
            override_data=self.args.override_data,
            model_types=test_model_types,
            sample_num=sample_num_te,
        )

        self.logger.info("Train data = {}, Test data = {}".format(len(train_set), len(test_set)))
        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)
        return train_loader, test_loader


    def init_train_test_dataset(self):
        train_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p, 
            override_data=self.args.override_data,
            train_test_stage="train",
            sample_num=self.args.train_num,
        )
        test_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p, 
            override_data=self.args.override_data,
            train_test_stage="test",
        )

        self.logger.info("Train data = {}, Test data = {}".format(
            len(train_set), len(test_set)))
        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)
        return train_loader, test_loader


    def init_model(self):
        best_acc = 1e9
        best_tau = -1e9
        start_epoch = 1

        if self.args.pretrain:
            self.logger.info("Loading pretrain: {}".format(self.args.pretrain))
            ckpt = torch.load(self.args.pretrain)
            self.model.load_state_dict(ckpt['state_dict'], strict = False)
            self.logger.info("Loaded pretrain: {}".format(self.args.pretrain))

        if self.args.resume:
            self.logger.info("Loading checkpoint: {}".format(self.args.resume))
            ckpt = torch.load(self.args.resume)
            start_epoch, best_acc, best_tau = ckpt['epoch'], ckpt['best_acc'], ckpt['best_tau']
            self.model.load_state_dict(ckpt['state_dict'], strict = True)
            self.logger.info("loaded checkpoint: {} (epoch {}  best acc{:.2f}  best tau{:.2f})". \
                             format(self.args.resume, start_epoch, best_acc, best_tau))

            self.logger.info("Loading optimizer and scheduler: {}".format(self.args.resume))
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.logger.info("Loaded optimizer and scheduler: {}".format(self.args.resume))

        self.best_acc, self.best_tau, self.start_epoch = best_acc, best_tau, start_epoch

    def format_second(self, secs):
        return "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format( \
               int(secs / 3600), int((secs % 3600) / 60), int(secs % 60))
    

    def save_checkpoint(self, epoch, best = False, latest = False):
        epoch_str = "best" if best else "e{}".format(epoch)
        if latest == True:
            epoch_str = "latest"
        model_path = "{}/ckpt_{}.pth".format(self.args.model_dir, epoch_str)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if latest:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_acc': self.best_acc,
                'best_tau': self.best_tau,
                }, model_path)
        else:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'best_tau': self.best_tau,
                }, model_path)
        self.logger.info("Checkpoint saved to {}".format(model_path))
        return

    def unpack_batch(self, batch):
        if self.args.dataset == 'nnlqp':
            data, static_feature, n_edges, _, plt_id = batch
            y = data.y.view(-1, 1).to(self.device)
            data = data.to(self.device)
            static_feature = static_feature.to(self.device)
            return data, static_feature, n_edges, y
        else:
            if not self.args.only_test and self.args.lambda_cons > 0:
                data1, data2 = batch
                code1, adj1, N1, V_A1, T_A1 = data1
                code2, adj2, N2, V_A2, T_A2 = data2
                code = torch.cat([code1, code2], dim=0).to(self.device)
                adj = torch.cat([adj1, adj2], dim=0).to(self.device)
                N = torch.cat([N1, N2], dim=0)
                V_A = torch.cat([V_A1, V_A2], dim=0)
            else:
                code, adj, N, V_A, T_A = batch
                code = code.to(self.device)
                adj = adj.to(self.device)

            if self.args.only_test:
                return code, adj, N, T_A.view(-1,1).to(self.device)
            else:
                return code, adj, N, V_A.view(-1,1).to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        metric = Metric()

        num_iter = len(self.train_loader)

        for iteration, batch in enumerate(self.train_loader):
            torch.cuda.empty_cache()

            data1, data2, n_edges, y = self.unpack_batch(batch)

            self.optimizer.zero_grad()
            
            # NNLQP: data1=data, data2=static feature
            # NasBench101/201: data1=netcode, data2=adjacency matrix
            pred_cost = self.model(data1, data2, n_edges)

            loss = F.mse_loss(pred_cost / y, y / y)
            if self.args.lambda_sr > 0:
                loss_sr = self.Loss_SR(pred_cost, y) * self.args.lambda_sr
                loss += loss_sr 
            if self.args.lambda_cons > 0:
                source_pre, auged_pre = torch.split(pred_cost, pred_cost.shape[0]//2, dim=0)
                loss_cons = F.l1_loss(auged_pre, source_pre) * self.args.lambda_cons
                loss += loss_cons
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            ps = pred_cost.data.cpu().numpy()[:, 0].tolist()
            gs = y.data.cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)

            if iteration % self.args.print_freq ==  0 or iteration+1 == num_iter:
                acc, err, tau = metric.get()
                t1 = time.time()
                speed = (t1 - t0) / (iteration + 1)
                exp_time = self.format_second(speed * (num_iter * (self.args.epochs - epoch + 1) - iteration))

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.logger.info("Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} MAPE:{:.5f} " \
                                "ErrBnd:{} Tau:{:.5f} Speed:{:.2f} ms/iter {}" .format( \
                                epoch, self.args.epochs, iteration, num_iter, lr, loss.data, acc, \
                                err, tau, speed * 1000, exp_time))
                if self.args.lambda_sr > 0 and self.args.lambda_cons > 0:
                    self.logger.info("Loss_SR:{:.5f} Loss_consistency:{:.5f}".format(loss_sr.data, loss_cons.data))
        return acc


    def test(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()
        t0 = time.time()
        num_iter = len(self.test_loader)
        if num_iter <= 0:
            return 0, 0
        metric = Metric()

        infer_time = 0
        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                torch.cuda.empty_cache()

                data1, data2, n_edges, y = self.unpack_batch(batch)

                # NNLQP: data1=data, data2=static feature
                # NasBench101/201: data1=netcode, data2=adjacency matrix
                time_i_1 = time.time()
                pred_cost = self.model(data1, data2, n_edges)
                time_i_2 = time.time()
                infer_time += time_i_2 - time_i_1

                # pred_cost = torch.exp(pred_cost)
                ps = pred_cost.data.cpu().numpy()[:, 0].tolist()
                gs = y.data.cpu().numpy()[:, 0].tolist()
                plts = None
                metric.update(ps, gs, plts)
                acc, err, tau = metric.get()

                if iteration > 0 and iteration % 50 == 0:
                    # self.logger.info("[{}/{}] MAPE: {:.5f} ErrBnd(0.1): {:.5f}".format(
                    self.logger.info("[{}/{}] MAPE: {:.5f} ErrBnd: {}, Tau: {:.5f}".format(
                        iteration, num_iter, acc, err, tau))

            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000
            acc, err, tau = metric.get()

            self.logger.info(" ------------------------------------------------------------------")
            self.logger.info(" * Speed: {:.5f} ms/iter".format(speed))
            self.logger.info(" * MAPE: {:.5f}".format(acc))
            self.logger.info(" * ErrorBound: {}".format(err))
            self.logger.info(" * Kendall's Tau: {}".format(tau))
            self.logger.info(" ------------------------------------------------------------------")

        if self.args.only_test:
                self.logger.info(" Average Latency : {:.8f} ms".format(infer_time/num_iter*1000))
                
        torch.manual_seed(time.time())
        torch.cuda.manual_seed_all(time.time())
        return acc, err, tau


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.args.only_test = False
            self.train_epoch(epoch)
            self.save_checkpoint(epoch, latest = True)
            
            if epoch > 0 and epoch % self.args.ckpt_save_freq == 0:
                self.save_checkpoint(epoch)

            if epoch > 0 and epoch % self.args.test_freq == 0:
                self.args.only_test = True
                acc, err, tau = self.test()

                if (self.args.dataset=='nnlqp' and acc < self.best_acc) or \
                   ('nasbench' in self.args.dataset and tau > self.best_tau):
                    self.best_acc = acc
                    self.best_err = err
                    self.best_tau = tau
                    self.save_checkpoint(epoch, best = True)
 
        self.logger.info("Train over, best acc = {:.5f}, err = {}, tau = {}".format(
            self.best_acc, self.best_err, self.best_tau
        ))
        return


    def run(self):
        if self.args.only_test:
            self.test()
        else:
            self.train()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
