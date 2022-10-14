from torch.nn import Parameter


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """
    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def dist(self, p1, p2, c):
        """Distance between a pair of points"""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    # the def defined by mine
    def l_inner(self, x, y, keep_dim):
        """Lorentz inner"""
        raise NotImplementedError

    def induced_distance(self, x, y, c):
        """Metric distance"""
        raise NotImplementedError

    def lorentzian_distance(self, x, y, c):
        """lorzentzian distance"""
        raise NotImplementedError

    def exp_map_x(self, p, dp, c, is_res_normalize, is_dp_normalize):
        raise NotImplementedError

    def exp_map_zero(self, dp, c, is_res_normalize, is_dp_normalize):
        raise NotImplementedError

    def log_map_x(self, x, y, c, is_tan_normalize):
        raise NotImplementedError

    def log_map_zero(self, y, c, is_tan_normalize):
        raise NotImplementedError

    def matvec_proj(self, m, x, c):
        raise NotImplementedError

    def matvecbias_proj(self, m, x, b, c):
        raise NotImplementedError

    def matvec_regular(self, m, x, c):
        raise NotImplementedError

    def matvecbias_regular(self, m, x, b, c):
        raise NotImplementedError

    def normalize_tangent_zero(self, p_tan, c):
        raise NotImplementedError

    def lorentz_centroid(self, weight, x, c):
        raise NotImplementedError

    def normalize_input(self, x, c):
        raise NotImplementedError

    def normlize_tangent_bias(self, x, c):
        raise NotImplementedError

    def proj_tan_zero(self, u, c):
        raise NotImplementedError

    def lorentz2poincare(self, x, c):
        raise NotImplementedError

    def poincare2lorentz(self, x, c):
        raise NotImplementedError

    def _lambda_x(self, x, c):
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        super().__init__(data, requires_grad)
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
