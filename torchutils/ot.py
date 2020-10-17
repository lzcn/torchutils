import numpy as np
import scipy as sp
import torch

from .ops import logmm


def sinkhorn_div(
    dist: torch.Tensor, a: torch.Tensor = None, b: torch.Tensor = None, v: torch.Tensor = None, gamma=0.01, step=10
):
    r"""Compute sinkhorn divergence and solve the entropic regularization optimal transport problem.

    The optimiazation problem is defined as follows:

    .. math::

        \begin{aligned}
        P_{\gamma} &=\underset{P\in \Pi(a,b)}{\text{argmin}}<P,M_{xy}>-\gamma E(P)\\
        s.t. &P_{\gamma} 1 = a \\
             &P_{\gamma}^\intercal 1= b \\
        \end{aligned}

    where :math:`M_{xy}` is the cost (distance) matrix over two set of points :math:`x\sim P_x,y\sim P_y`,
    :math:`a,b` is the distribution of :math:`x,y` and :math:`E(P)` is the entropic regularization.

    The solver iterates over the following two steps:

    .. math::

        \begin{equation}
        \left\{\begin{aligned}
        u&=a / K v \\
        v&=b / K^\intercal u
        \end{aligned}\right.
        \end{equation}

    where :math:`P_{\gamma}=\text{diag}(u)K\text{diag}(v)` and :math:`K=\exp{(-M_{xy}/\gamma})`


    Args:
        dist (torch.Tensor): the cost (distance) matrix :math:`M_{xy}`.
        a (torch.Tensor, optional): Dirichlet distribution over :math:`x`. Defaults to None (uniform).
        b (torch.Tensor, optional): Dirichlet distribution over :math:`y`. Defaults to None (uniform).
        v (torch.Tensor, optional): initial state of :math:`v` . Defaults to None.
        gamma (float, optional): regularization strength. Defaults to 0.01.
        step (int, optional): number of iterations. Defaults to 10.

    """
    n, m = dist.shape
    K = torch.exp(-dist / gamma)
    a = dist.new_ones(n) / n if a is None else a
    b = dist.new_ones(m) / m if b is None else b
    v = dist.new_ones(m) if v is None else v

    for i in range(step):
        u = a / torch.matmul(K, v)
        v = b / torch.matmul(u, K)
    P = u[:, None] * K * v[None, :]
    loss = (P * dist).sum()
    return loss, P, u, v


def sinkhorn_div_stable(
    dist: torch.Tensor, a: torch.Tensor = None, b: torch.Tensor = None, v: torch.Tensor = None, gamma=0.01, step=10
):
    n, m = dist.shape
    log_K = -dist / gamma
    log_a = torch.log(dist.new_ones(n) / n) if a is None else torch.log(a)
    log_b = torch.log(dist.new_ones(m) / n) if b is None else torch.log(b)
    log_v = dist.new_zeros(m) if v is None else torch.log(v)
    for _ in range(step):
        log_u = log_a - logmm(log_K, log_v[:, None]).view(-1)
        log_v = log_b - logmm(log_u[None, :], log_K).view(-1)

    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = torch.exp(log_P)
    loss = (P * dist).sum()

    return loss, P, torch.exp(log_u), torch.exp(log_v)
