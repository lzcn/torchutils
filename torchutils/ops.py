from math import log, pi

import numpy as np
import torch
import torch.nn as nn


def smooth(xs, win_size=10):
    """Average smooth for 1d signal."""
    assert win_size > 0, "win_size should be positive."
    if win_size == 1:
        return np.array(xs)
    if len(xs) < win_size:
        return _smooth(xs, win_size)
    weights = np.ones(win_size) / win_size
    data = np.convolve(xs, weights, mode="valid")
    pre = _smooth(xs[: win_size - 1], win_size)
    return np.hstack((pre, data))


def _smooth(xs, win_size=10):
    """Slower version for smooth."""
    x_buffer = []
    s_xs = 0
    xs = np.array(xs) * 1.0
    smoothed_xs = np.zeros_like(xs)
    num = xs.size
    for i in range(num):
        x = xs[i]
        if len(x_buffer) < win_size:
            x_buffer.append(x)
            size = len(x_buffer)
            s_xs = (s_xs * (size - 1) + x) / size
            smoothed_xs[i] = s_xs
        else:
            idx = i % win_size
            s_xs += (x - x_buffer[idx]) / win_size
            x_buffer[idx] = x
            smoothed_xs[i] = s_xs
    return smoothed_xs


def glorot_uniform(t, gain):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m, gain):
    if isinstance(m, nn.Parameter):
        glorot_uniform(m.data, gain)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data, gain)


def weights_init(m, gain=1.0):
    for module in m.modules():
        if isinstance(module, nn.ParameterList):
            for param in module:
                _param_init(param, gain)
        else:
            _param_init(module, gain)

    for name, module in m.named_parameters():
        if "." not in name:  # top-level parameters
            _param_init(module, gain)


class SPMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, _ = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None
        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = torch.mm(sp_mat.data.t(), grad_output.data)
        return grad_matrix1, grad_matrix2


def gnn_spmm(sp_mat, dense_mat):
    return SPMM.apply(sp_mat, dense_mat)


def convert_to_int(x, bits=8):
    """Convert binary codes x into a set of int8.

    Parameters
    ----------
    x: size of n, with binary value in {+1,-1}
    Return:
    ------
    y: size of ceil(n/8), with each one being the int8 of 8 bits
    """
    size = len(x) // bits
    y = []
    x = np.maximum(x, 0)
    for n in range(size):
        y.append(int("".join(map(str, x[n * bits : (n + 1) * bits])), 2))
    return np.array(y)


def convert_to_binary(v, bits=8):
    x = "{:b}".format(v).zfill(bits)
    x = np.array(list(map(int, x)))
    return x


def _build_one_table(weights, bits=8):
    table = []
    for i in range(2 ** bits):
        x = convert_to_binary(i, bits)
        # 0 -> 1 , 1 -> -1
        x = 1 - 2 * x
        table.append(np.sum(x * weights))
    return np.array(table)


def build_look_table(weights, bits=8):
    size = len(weights) // bits
    tables = []
    for n in range(size):
        tables.append(_build_one_table(weights[n * bits : (n + 1) * bits], bits))
    return np.array(tables)


def hamming_sim(xs, ys, tables, offset):
    return tables[np.bitwise_xor(xs, ys) + offset].sum()


def test_whd(size=600, bits=8):
    num = size * bits
    a = np.random.choice([-1, 1], num)
    b = np.random.choice([-1, 1], num)
    w = np.random.randn(num)
    x = convert_to_int(a, bits)
    y = convert_to_int(b, bits)
    tables = build_look_table(w, bits).reshape(size * (2 ** bits))
    offset = np.arange(size) * (2 ** bits)
    sim1 = np.sum(w * a * b)
    sim2 = tables[np.bitwise_xor(x, y) + offset].sum()
    assert np.abs(sim1 - sim2) < 1e-6


def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2 * pi)
    b = logvar
    c = torch.exp(-logvar) * (x - mean) ** 2
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    return 0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1 - logvar)


def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = torch.exp(-p_logvar) * torch.exp(q_logvar)
    b = torch.exp(-p_logvar) * ((q_mean - p_mean) ** 2)
    c = p_logvar - q_logvar
    return 0.5 * torch.sum(a + b - 1 + c)


def logmm(logx: torch.Tensor, logy: torch.Tensor):
    r"""Performs a matrix multiplication of the matrices ``logx`` and ``logy`` with logarithm values.

    ``logx`` is a :math:`(n\times m)` tensor, ``logy`` is a :math:`(m\times p)` tensor
    and ``out`` is a :math:`(n\times p)` tensor.

    Args:
        logx (torch.Tensor): the first matrix.
        logy (torch.Tensor): the second matrix.

    Returns:
        torch.Tensor: output matrix

    Note:
        This function does not broadcast.

    """
    # n x m x 1
    logx = logx.unsqueeze(-1)
    # 1 x m x p
    logy = logy.unsqueeze(0)
    logk = logx + logy
    logm, _ = logk.max(dim=1, keepdim=True)
    logz = torch.log(torch.exp(logk - logm).sum(dim=1)) + logm.sum(dim=1)
    return logz


def pairwise_distances(x: torch.Tensor, y: torch.Tensor):
    r"""Computes (batched) pairwise distance with mm.

    Args:
        x (torch.Tensor): input tensor of shape :math:`(B\times N \times D)` or :math:`(N \times D)`
        y (torch.Tensor): input tensor of shape :math:`(B\times M \times D)` or :math:`(M \times D)`

    The output is tensor of shape :math:`(B\times N\times M)` or :math:`(N\times M)`.

    Note:
        This implementation is similar to :meth:`~torch.cdist` with default setting.
    """
    x_norm = (x * x).sum(dim=-1, keepdim=True)
    y_norm = (y * y).sum(dim=-1, keepdim=True).transpose(-1, -2)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, torch.transpose(y, -1, -2))
    return dist ** 0.5


def kaiming_uniform(module, nonlinearity="relu"):
    """Kaiming uniform for neural networks.

    Args:
        module (nn.Module): module to apply the initialization
        nonlinearity (str, optional): non-linearity. Defaults to "relu".
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
