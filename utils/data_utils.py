import os
import numpy as np
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
from torch_geometric.utils import remove_self_loops

from utils.util import logger


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def loader(args):
    filepath = os.path.join(args.data_pt_path, '{}.data'.format(args.dataset))
    if os.path.isfile(filepath):
        logger.info('loading {} ...'.format(args.dataset))
        return torch.load(filepath)
    else:
        logger.info('{} does not exist, exiting...')
        return None


def prepare_dilated_edge_index(data, spatial_dilated_factors, device):
    dilated_edge_index_list = []
    logger.info('computing spatial dilated edge list ...')
    for edge_index in tqdm(data['edge_index_list']):
        adj = coo_matrix(([1] * len(edge_index[0]), (list(edge_index[0]), list(edge_index[1]))), dtype=int)
        adj = adj.tocsr() + adj.transpose().tocsr()
        dilated_edge_index = []
        for factor in spatial_dilated_factors:
            exponent = 1
            adj_exp = adj
            while exponent < factor:
                adj_exp = adj_exp.dot(adj)
                exponent += 1
            adj_exp = adj_exp.tocoo()
            adj_exp.eliminate_zeros()
            coords = np.vstack((adj_exp.row, adj_exp.col)).transpose()
            np.random.shuffle(coords)
            coords, _ = remove_self_loops(torch.tensor(coords.transpose(), dtype=torch.long))
            dilated_edge_index.append(coords.to(device))
        dilated_edge_index_list.append(dilated_edge_index)
    return dilated_edge_index_list


def prepare_train_test_data(data, t, device):
    edge_index = data['edge_index_list'][t].long().to(device)
    pos_index = data['pedges'][t].long().to(device)
    neg_index = data['nedges'][t].long().to(device)
    new_pos_index = data['new_pedges'][t].long().to(device)
    new_neg_index = data['new_nedges'][t].long().to(device)
    return edge_index, pos_index, neg_index, new_pos_index, new_neg_index
