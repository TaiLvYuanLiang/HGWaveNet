import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot

from model.spatial_dilated_conv import SpatialDilatedConv
from model.temporal_casual_conv import TemporalCasualConv
from layers import HypLinear
from manifolds import Euclidean, Lorentzian, PoincareBall


class HGWaveNet(nn.Module):
    def __init__(self, args):
        super(HGWaveNet, self).__init__()
        if args.manifold == 'PoincareBall':
            self.manifold = PoincareBall()
        elif args.manifold == 'Lorentzian':
            self.manifold = Lorentzian()
        elif args.manifold == 'Euclidean':
            self.manifold = Euclidean()
        else:
            raise RuntimeError('invalid argument: manifold')
        self.device = args.device
        self.window_size = args.casual_conv_kernel_size ** args.casual_conv_depth
        self.history_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        self.history = []
        self.spatial_dilated_conv = SpatialDilatedConv(args)
        self.c_out = self.spatial_dilated_conv.c_out
        self.nout = args.nout
        self.temporal_casual_conv = TemporalCasualConv(args, self.c_out)
        self.gru = nn.GRUCell(args.nout, args.nout)
        self.linear = HypLinear(self.manifold, args.nout, args.nout, self.c_out, self.device,
                                dropout=args.dropout, use_bias=args.bias)
        self.Q = Parameter(torch.ones((args.nout, args.nhid)).to(args.device), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)).to(args.device), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.history_initial)
        glorot(self.Q)
        glorot(self.r)

    def to_hyper(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def to_tangent(self, x, c=1.0):
        x_tan = self.manifold.logmap0(x, c)
        x_tan = self.manifold.proj_tan0(x_tan, c)
        return x_tan

    def init_hyper(self, x, c=1.0):
        if isinstance(self.manifold, Lorentzian):
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        return self.to_hyper(x, c)

    def init_history(self):
        self.history = [self.init_hyper(self.history_initial).unsqueeze(0)] * self.window_size
        return self.history

    def aggregate_history(self, history):
        att = torch.matmul(torch.tanh(torch.matmul(history, self.Q)), self.r)
        att = torch.reshape(att, (self.window_size, -1))
        att = F.softmax(att, dim=0).unsqueeze(2)
        history_reshape = torch.reshape(history, [self.window_size, -1, self.nout])
        history_agg = torch.mean(att * history_reshape, dim=0)
        return history_agg

    def htc(self, x):
        x = self.manifold.proj(x, self.c_out)
        h = self.manifold.proj(self.history[-1], self.c_out)
        return self.manifold.sqdist(x, h, self.c_out).squeeze().mean()

    def update_history(self, x):
        self.history.pop(0)
        self.history.append(x.clone().detach().requires_grad_(False).unsqueeze(0))

    def forward(self, dilated_edge_index, x=None):
        x = self.spatial_dilated_conv(dilated_edge_index, x)
        history = torch.cat(self.history, dim=0)
        x = self.gru(self.to_tangent(x, self.c_out),
                     self.to_tangent(self.temporal_casual_conv(history)[-1], self.c_out))
        x = self.to_hyper(x, self.c_out)
        x = self.linear(x)
        return x
