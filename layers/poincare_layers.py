import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros


class HGATConv(nn.Module):
    """
    Poincare graph convolution layer.
    """
    def __init__(self, manifold, in_features, out_features, c_in, c_out, device, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, device, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, device, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.device = device

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv(nn.Module):
    """
    Poincare graph convolution layer, from HGCN。
    """
    def __init__(self, manifold, in_features, out_features, device, c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, device, dropout=dropout, use_bias=use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, device, bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in
        self.device = device

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Poincare linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c, device, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.device = device
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAttAgg(MessagePassing):
    def __init__(self, manifold, c, out_features, device, att_dropout=0.6, heads=1, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.device = device
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels).to(device), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels).to(device), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return support_t


class HypAct(Module):
    """
    Poincare activation layer.
    """
    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAgg(MessagePassing):
    """
    Poincare aggregation layer using degree.
    """
    def __init__(self, manifold, c, out_features, device, bias=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.device = device
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1).to(device))

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_j = torch.nn.functional.embedding(node_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypConv1d(nn.Module):
    def __init__(self, manifold, in_size, out_size, kernel_size, c, device, dilation=1, stride=1):
        super(HypConv1d, self).__init__()
        self.manifold = manifold
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.c = c
        self.device = device
        self.dilation = dilation
        self.stride = stride
        self.pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad,
                              stride=stride, dilation=dilation, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.conv.weight)

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
        x = self.to_tangent(self.manifold.proj(x, self.c), self.c)
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x = self.to_hyper(x, self.c)
        return x
