from typing import Tuple
import torch

from .ops import logmm


def sinkhorn(
    M: torch.Tensor,
    a: torch.Tensor = None,
    b: torch.Tensor = None,
    v: torch.Tensor = None,
    gamma: float = 0.01,
    budget: int = 10,
    stable: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        M (torch.Tensor): the cost (distance) matrix :math:`M_{xy}`.
        a (torch.Tensor, optional): Dirichlet distribution over :math:`x`. Defaults to None (uniform).
        b (torch.Tensor, optional): Dirichlet distribution over :math:`y`. Defaults to None (uniform).
        v (torch.Tensor, optional): initial state of :math:`v` . Defaults to None.
        gamma (float, optional): regularization strength. Defaults to 0.01.
        budget (int, optional): number of iterations. Defaults to 10.
        stable (bool, optional): whether to use stable version.

    """
    if stable:
        return sinkhorn_div_stable(M, a, b, v, gamma, budget)
    else:
        return sinkhorn_div(M, a, b, v, gamma, budget)


def sinkhorn_div(
    M: torch.Tensor,
    a: torch.Tensor = None,
    b: torch.Tensor = None,
    v: torch.Tensor = None,
    gamma: float = 0.01,
    budget: int = 10,
):
    n, m = M.shape
    K = torch.exp(-M / gamma)
    a = M.new_ones(n) / n if a is None else a
    b = M.new_ones(m) / m if b is None else b
    v = M.new_ones(m) if v is None else v
    u = M.new_ones(n)

    for _ in range(budget):
        u = a / torch.matmul(K, v)
        v = b / torch.matmul(u, K)
    P = u[:, None] * K * v[None, :]
    loss = (P * M).sum()
    return loss, P, u, v


def sinkhorn_div_stable(
    M: torch.Tensor,
    a: torch.Tensor = None,
    b: torch.Tensor = None,
    v: torch.Tensor = None,
    gamma: float = 0.01,
    budget: int = 10,
):
    n, m = M.shape
    log_K = -M / gamma
    log_a = torch.log(M.new_ones(n) / n) if a is None else torch.log(a)
    log_b = torch.log(M.new_ones(m) / n) if b is None else torch.log(b)
    log_v = M.new_zeros(m) if v is None else torch.log(v)
    log_u = M.new_zeros(n)

    for _ in range(budget):
        log_u = log_a - logmm(log_K, log_v[:, None]).view(-1)
        log_v = log_b - logmm(log_u[None, :], log_K).view(-1)

    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = torch.exp(log_P)
    loss = (P * M).sum()

    return loss, P, torch.exp(log_u), torch.exp(log_v)
