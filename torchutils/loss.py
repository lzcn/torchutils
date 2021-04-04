import torch
import torch.nn.functional as F


def contrastive_loss(s_data: torch.Tensor, t_data: torch.Tensor, margin=0.2, mask=None, reduction="mean"):
    r"""Compute the contrastive loss for two domains as follows:

    .. math::

        \ell(t,s) = \frac{1}{n-1}\left(
            \sum_{i}^{n}\max(0, m - \text{sim}(t,s) + \text{sim}(t,s_i)) +
            \sum_{j}^{n}\max(0, m - \text{sim}(t,s) + \text{sim}(f_j,s))\right)

    where :math:`t` and :math:`s` are the features of different domains of the same
    sample and :math:`f_i` and :math:`s_j` are the features of different samples.

    We compute the similarity using cosine similarity.

    Todo:
        to support more similarity.

    Args:
        s_data (torch.Tensor) : array of shape :math:`(N, D)`, source features
        t_data (torch.Tensor) : array of shape :math:`(N, D)`, target features
        margin (float, optional): the margin :math:`m`. Defaults to 0.1.
        mask (None, optional): array of shape :math:`(N)`. mask for samples.
        reduction (str, optional): Defaults to "mean". Specifies the reduction to
            apply to the output:

            - `'none'`: no reduction will be applied,
            - `'mean'`: the sum of the output will be divided by the number of elements in the output,
            - `'sum'`: the output will be summed.

    Note:
        If mask is not None, the output with "none" reduction may contains zeros.
        Such that simply averaging the output will give a smaller result.

    Returns:
        torch.Tensor: shape :math:`(N)` or :math:`(1)`

    """
    # compute the cosine similarity
    assert s_data.size(0) == t_data.size(0)
    s_data = F.normalize(s_data, dim=-1)
    t_data = F.normalize(t_data, dim=-1)
    sim = s_data.matmul(t_data.t())
    # d(f,v)
    diag = sim.diag()
    zeros = torch.zeros_like(sim)
    # sum along the row to get the VSE loss for each image
    s_cost = torch.max(zeros, margin - diag.view(-1, 1) + sim)
    # sum along the column to get the VSE loss for each sentence
    t_cost = torch.max(zeros, margin - diag.view(1, -1) + sim)
    if mask is not None:
        mask = mask.view(-1, 1) * 1.0
        mat_mask = mask.matmul(mask.t()) > 0
        s_cost = s_cost.masked_fill_(~mat_mask, 0.0)
        t_cost = t_cost.masked_fill_(~mat_mask, 0.0)
    else:
        mask = torch.ones_like(diag).view(-1, 1)
    # n x 1
    loss = s_cost.sum(dim=1) + t_cost.sum(dim=0)
    # normalized by the number of items
    loss = loss / (mask.sum() - 1.0)
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.sum() / mask.sum()
    else:
        raise KeyError


def soft_margin_loss(pos, neg=None, reduction="none"):
    """Compute the BPR loss."""
    if neg is None:
        x = pos
    else:
        x = pos - neg
    return F.soft_margin_loss(x, torch.ones_like(x), reduction=reduction)
