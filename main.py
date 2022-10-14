import torch
import numpy as np
import time
import geoopt
import networkx as nx
from math import isnan
from tqdm import tqdm

from config import args
from utils.data_utils import loader, prepare_dilated_edge_index, prepare_train_test_data
from utils.util import set_random, logger, hyperbolicity_sample
from model import HGWaveNet
from loss import ReconLoss


class Trainer(object):
    def __init__(self):
        self.data = loader(args)
        if self.data is None:
            raise RuntimeError('dataset not exsits')
        args.num_nodes = self.data['num_nodes']
        self.train_shots = list(range(0, self.data['time_length'] - args.test_length))
        self.test_shots = list(range(self.data['time_length'] - args.test_length, self.data['time_length']))
        self.model = HGWaveNet(args).to(args.device)
        self.loss = ReconLoss(args, self.model.c_out)
        if args.use_riemannian_adam:
            self.optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                               weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        set_random(args.seed)

    def train(self):
        dilated_edge_index_list = prepare_dilated_edge_index(self.data, args.spatial_dilated_factors, args.device)
        t_total = time.time()
        test_result = []
        min_loss = 1.0e8
        patience = 0
        for epoch in range(1, args.max_epoch + 1):
            t_epoch = time.time()
            epoch_losses = []
            z = None
            epoch_loss = None
            self.model.init_history()
            self.model.train()
            for t in self.train_shots:
                edge_index, _, _, _, _ = prepare_train_test_data(self.data,
                                                                 t if (t + 1) not in self.train_shots else (t + 1),
                                                                 args.device)
                dilated_edge_index = dilated_edge_index_list[t]
                self.optimizer.zero_grad()
                z = self.model(dilated_edge_index)
                epoch_loss = self.loss(z, edge_index) + self.model.htc(z)
                epoch_loss.backward()
                if isnan(epoch_loss):
                    logger.info('==' * 25)
                    logger.info('nan loss')
                    break
                self.optimizer.step()
                epoch_losses.append(epoch_loss.item())
                self.model.update_history(z)
            if isnan(epoch_loss):
                break
            self.model.eval()
            average_epoch_loss = np.mean(epoch_losses)
            train_result = self.test(z, is_training=True)
            test_result = self.test(z)
            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                patience = 0
            else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:
                    logger.info('==' * 25)
                    logger.info('early stopping!')
                    break
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 25)
                logger.info('Epoch:{}, Loss:{:.4f}, Time:{:.3f}'
                            .format(epoch,
                                    average_epoch_loss,
                                    time.time() - t_epoch))
                logger.info('Epoch:{}, Train, AUC:{:.4f}, AP:{:.4f}, new AUC:{:.4f}, new AP:{:.4f}'
                            .format(epoch,
                                    train_result[0],
                                    train_result[1],
                                    train_result[2],
                                    train_result[3]))
                logger.info('Epoch:{}, Test, AUC:{:.4f}, AP:{:.4f}, new AUC:{:.4f}, new AP:{:.4f}'
                            .format(epoch,
                                    test_result[0],
                                    test_result[1],
                                    test_result[2],
                                    test_result[3]))
        logger.info('==' * 25)
        logger.info('Total time: {:.3f}'.format(time.time() - t_total))
        return test_result

    def test(self, embeddings, is_training=False):
        auc_list, ap_list, new_auc_list, new_ap_list = [], [], [], []
        embeddings = embeddings.detach()
        shots = [self.train_shots[-1]] if is_training else self.test_shots
        for t in shots:
            _, pos_index, neg_index, new_pos_index, new_neg_index = \
                prepare_train_test_data(self.data, t, args.device)
            auc, ap = self.loss.predict(embeddings, pos_index, neg_index)
            new_auc, new_ap = self.loss.predict(embeddings, new_pos_index, new_neg_index)
            auc_list.append(auc)
            ap_list.append(ap)
            new_auc_list.append(new_auc)
            new_ap_list.append(new_ap)
        return np.mean(auc_list), np.mean(ap_list), np.mean(new_auc_list), np.mean(new_ap_list)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    