import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from manifolds import Euclidean, Lorentzian, PoincareBall


class ReconLoss(nn.Module):
    def __init__(self, args, c):
        super(ReconLoss, self).__init__()
        if args.manifold == 'PoincareBall':
            self.manifold = PoincareBall()
        elif args.manifold == 'Lorentzian':
            self.manifold = Lorentzian()
        elif args.manifold == 'Euclidean':
            self.manifold = Euclidean()
        else:
            raise RuntimeError('invalid argument: manifold')
        self.device = args.device
        self.c = c
        self.eps = args.eps
        self.negative_sampling = negative_sampling
        self.sampling_times = args.sampling_times
        self.fermidirac_decoder = FermiDiracDecoder(2.0, 1.0)
        self.use_hyperdecoder = args.use_hyperdecoder and (not isinstance(self.manifold, Euclidean))

    @staticmethod
    def decoder(z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def hyperdecoder(self, z, edge_index):
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = F.embedding(edge_i, z)
        z_j = F.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, self.c).squeeze()
        return self.fermidirac_decoder(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        decoder = self.hyperdecoder if self.use_hyperdecoder else self.decoder
        pos_loss = -torch.log(decoder(z, pos_edge_index) + self.eps).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=pos_edge_index.size(1) * self.sampling_times)
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + self.eps).mean()
        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdecoder if self.use_hyperdecoder else self.decoder
        pos_y = z.new_ones(pos_edge_index.size(1)).to(self.device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(self.device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs
