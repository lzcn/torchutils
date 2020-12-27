import torch
import torch.nn.functional as F

from .ops import logmm


def contrastive_loss(im, s, margin=0.1, norm=False, reduction="none"):
    r"""Compute the contrastive loss for two modalities as follows:

    .. math::

        \ell(f,s) =
            \sum_{i}\max(0, m - \text{sim}(f,s) + \text{sim}(f,s_i)) +
            \sum_{j}\max(0, m - \text{sim}(f,s) + \text{sim}(f_j,s))

    where :math:`f` and :math:`s` are the features for different modaliteis of the same
    sample and :math:`f_i` and :math:`s_j` are the features of different samples.

    Args:
        im (torch.Tensor) : array of shape :math:`(N, D)`, image features
        s (torch.Tensor) : array of shape :math:`(N, D)`, sentence features
        margin (float, optional): margin, by default 0.1
        norm (bool, optional): optional, whether to normarlize features
        reduction (str, optional): same as :class:`torch.nn.L1Loss`

    Returns:
        torch.Tensor: shape :math:`(N, 1)`, contrastive loss between two modalities.

    """
    size, dim = im.shape
    if norm:
        im = im / im.norm(dim=1, keepdim=True)
        s = s / s.norm(dim=1, keepdim=True)
    scores = im.matmul(s.t()) / dim
    diag = scores.diag()
    zeros = torch.zeros_like(scores)
    # shape #item x #item
    # sum along the row to get the VSE loss from each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + scores)
    # sum along the column to get the VSE loss from each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + scores)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0) - 2 * margin
    # normalized by the number of items
    vse_loss = vse_loss / (size - 1)
    if reduction == "none":
        return vse_loss
    elif reduction == "sum":
        return vse_loss.sum()
    elif reduction == "mean":
        return vse_loss.mean()
    else:
        raise KeyError


def soft_margin_loss(pos, neg=None, reduction="none"):
    """Compute the BPR loss."""
    if neg is None:
        x = pos
    else:
        x = pos - neg
    return F.soft_margin_loss(x, torch.ones_like(x), reduction=reduction)


def sinkhorn_div(
    distance: torch.Tensor,
    a: torch.Tensor = None,
    b: torch.Tensor = None,
    v: torch.Tensor = None,
    gamma=0.01,
    budget=10,
):
    n, m = distance.shape
    K = torch.exp(-distance / gamma)
    a = distance.new_ones(n) / n if a is None else a
    b = distance.new_ones(m) / m if b is None else b
    v = distance.new_ones(m) if v is None else v
    u = distance.new_ones(n)

    for _ in range(budget):
        u = a / torch.matmul(K, v)
        v = b / torch.matmul(u, K)
    P = u[:, None] * K * v[None, :]
    loss = (P * distance).sum()
    return loss, P, u, v


def sinkhorn_div_stable(
    distance: torch.Tensor,
    a: torch.Tensor = None,
    b: torch.Tensor = None,
    v: torch.Tensor = None,
    gamma=0.01,
    budget=10,
):
    n, m = distance.shape
    log_K = -distance / gamma
    log_a = torch.log(distance.new_ones(n) / n) if a is None else torch.log(a)
    log_b = torch.log(distance.new_ones(m) / n) if b is None else torch.log(b)
    log_v = distance.new_zeros(m) if v is None else torch.log(v)
    log_u = distance.new_zeros(n)

    for _ in range(budget):
        log_u = log_a - logmm(log_K, log_v[:, None]).view(-1)
        log_v = log_b - logmm(log_u[None, :], log_K).view(-1)

    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = torch.exp(log_P)
    loss = (P * distance).sum()

    return loss, P, torch.exp(log_u), torch.exp(log_v)
