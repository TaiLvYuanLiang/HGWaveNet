import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot

from manifolds import Euclidean, Lorentzian, PoincareBall
from layers import HypConv1d


class TemporalCasualConv(nn.Module):
    def __init__(self, args, c):
        super(TemporalCasualConv, self).__init__()
        if args.manifold == 'PoincareBall':
            self.manifold = PoincareBall()
        elif args.manifold == 'Lorentzian':
            self.manifold = Lorentzian()
        elif args.manifold == 'Euclidean':
            self.manifold = Euclidean()
        else:
            raise RuntimeError('invalid argument: manifold')
        self.c = c
        self.device = args.device
        self.nout = args.nout
        self.residual_size = args.nout
        self.skip_size = args.nout
        self.casual_conv_depth = args.casual_conv_depth
        self.casual_conv_kernel_size = args.casual_conv_kernel_size
        self.dilated_stack = nn.ModuleList(
            [ResidualLayer(self.manifold, self.residual_size, self.skip_size, self.casual_conv_kernel_size,
                           self.casual_conv_kernel_size ** layer, self.c, device=self.device)
             for layer in range(self.casual_conv_depth)])

    def forward(self, x):
        skips = []
        for layer in self.dilated_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
        out = torch.cat(skips, dim=0).mean(dim=0)
        out = self.manifold.proj(out, self.c)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, manifold, residual_size, skip_size, kernel_size, dilation, c, device):
        super(ResidualLayer, self).__init__()
        self.manifold = manifold
        self.c = c
        self.device = device
        self.residual_size = residual_size
        self.skip_size = skip_size
        self.kernel_size = kernel_size * 2 - 1
        self.dilation = dilation
        self.conv_filter = HypConv1d(manifold, residual_size, residual_size, self.kernel_size,
                                     c=c, device=device, dilation=dilation)
        self.conv_gate = HypConv1d(manifold, residual_size, residual_size, self.kernel_size,
                                   c=c, device=device, dilation=dilation)
        self.conv_res = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_size, skip_size, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.conv_res.weight)
        glorot(self.conv_skip.weight)

    def to_tangent(self, x, c=1.0):
        x_tan = self.manifold.logmap0(x, c)
        x_tan = self.manifold.proj_tan0(x_tan, c)
        return x_tan

    def to_hyper(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        conv_filter = self.to_tangent(self.manifold.proj(conv_filter, self.c), self.c)
        conv_filter = self.to_hyper(torch.tanh(conv_filter), self.c)
        conv_gate = self.to_tangent(self.manifold.proj(conv_gate, self.c), self.c)
        fx = conv_filter * torch.sigmoid(conv_gate)
        fx = self.to_tangent(self.manifold.proj(fx, self.c), self.c)
        fx = fx.permute(1, 2, 0)
        fx = self.conv_res(fx)
        fx = fx.permute(2, 0, 1)
        fx = self.to_hyper(fx, self.c)
        skip = self.to_tangent(self.manifold.proj(fx, self.c), self.c)
        skip = skip.permute(1, 2, 0)
        skip = self.conv_skip(skip)
        skip = skip.permute(2, 0, 1)
        skip = self.to_hyper(skip, self.c)
        residual = self.manifold.mobius_add(fx, x, self.c)
        return skip, residual
