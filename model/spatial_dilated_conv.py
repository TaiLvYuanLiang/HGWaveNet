import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot

from manifolds import Euclidean, Lorentzian, PoincareBall
from layers import HGCNConv, HGATConv


class SpatialDilatedConv(nn.Module):
    def __init__(self, args):
        super(SpatialDilatedConv, self).__init__()
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nhid).to(args.device)
        if args.manifold == 'PoincareBall':
            self.manifold = PoincareBall()
        elif args.manifold == 'Lorentzian':
            self.manifold = Lorentzian()
        elif args.manifold == 'Euclidean':
            self.manifold = Euclidean()
        else:
            raise RuntimeError('invalid argument: manifold')
        self.c = Parameter(torch.ones(len(args.spatial_dilated_factors) * 3 + 1, 1).to(args.device) * args.curvature,
                           requires_grad=args.trainable_curvature)
        self.c_out = self.c[-1]
        self.spatial_layers = []
        if args.aggregation == 'deg':
            for i in range(len(args.spatial_dilated_factors)):
                layer1 = HGCNConv(self.manifold, args.nhid, args.nhid, args.device, self.c[i * 3],
                                  self.c[i * 3 + 1], dropout=args.dropout, use_bias=args.bias)
                layer2 = HGCNConv(self.manifold, args.nhid, args.nout, args.device, self.c[i * 3 + 1],
                                  self.c[i * 3 + 2], dropout=args.dropout, use_bias=args.bias)
                self.spatial_layers.append([layer1, layer2])
        elif args.aggregation == 'att':
            for i in range(len(args.spatial_dilated_factors)):
                layer1 = HGATConv(self.manifold, args.nhid, args.nhid, self.c[i * 3],
                                  self.c[i * 3 + 1], args.device, heads=args.heads, dropout=args.dropout,
                                  use_bias=args.bias, att_dropout=args.dropout, concat=True)
                layer2 = HGATConv(self.manifold, args.nhid * args.heads, args.nout, self.c[i * 3 + 1],
                                  self.c[i * 3 + 2], args.device, heads=args.heads, dropout=args.dropout,
                                  use_bias=args.bias, att_dropout=args.dropout, concat=False)
                self.spatial_layers.append([layer1, layer2])
        else:
            raise RuntimeError('invalid argument: aggregation')
        self.nhid = args.nhid
        self.nout = args.nout
        self.spatial_dilated_factors = args.spatial_dilated_factors
        self.Q = Parameter(torch.ones((args.nout, args.nhid)).to(args.device), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)).to(args.device), requires_grad=True)
        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.feat)
        glorot(self.linear.weight)
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

    def aggregate_dilated(self, dilated):
        att = torch.matmul(torch.tanh(torch.matmul(dilated, self.Q)), self.r)
        att = torch.reshape(att, (len(self.spatial_dilated_factors), -1))
        att = F.softmax(att, dim=0).unsqueeze(2)
        dilated_reshape = torch.reshape(dilated, [len(self.spatial_dilated_factors), -1, self.nout])
        dilated_agg = torch.mean(att * dilated_reshape, dim=0)
        return dilated_agg

    def forward(self, dilated_edge_index, x=None):
        x_list = []
        if x is None:
            x = self.linear(self.feat)
        else:
            x = self.linear(x)
        for i in range(len(self.spatial_dilated_factors)):
            x_f = self.init_hyper(x, self.c[i * 3])
            x_f = self.manifold.proj(x_f, self.c[i * 3])
            x_f = self.spatial_layers[i][0](x_f, dilated_edge_index[i])
            x_f = self.manifold.proj(x_f, self.c[i * 3 + 1])
            x_f = self.spatial_layers[i][1](x_f, dilated_edge_index[i])
            x_list.append(x_f)
        x = torch.cat([self.to_tangent(x_list[i], self.c[i * 3 + 2])
                       for i in range(len(self.spatial_dilated_factors))], dim=0)
        x = self.aggregate_dilated(x)
        x = self.to_hyper(x, self.c_out)
        return x
