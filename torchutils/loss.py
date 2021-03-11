import torch
import torch.nn.functional as F


def contrastive_loss(im: torch.Tensor, s: torch.Tensor, margin=0.2, mask=None, reduction="mean"):
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
        mask (None, optional): mask for tensor.
        reduction (str, optional): same as :class:`torch.nn.L1Loss`

    Returns:
        torch.Tensor: shape :math:`(N, 1)`, contrastive loss between two modalities.

    """
    # compute the cosine similarity
    im = F.normalize(im, dim=1)
    s = F.normalize(s, dim=-1)
    sim = im.matmul(s.t())
    # d(f,v)
    diag = sim.diag()
    zeros = torch.zeros_like(sim)
    # sum along the row to get the VSE loss for each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + sim)
    # sum along the column to get the VSE loss for each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + sim)
    if mask is not None:
        num = mask.sum()
        mask = mask.matmul(mask.t()) > 0
        cost_im = cost_im.masked_fill_(~mask, 0.0)
        cost_s = cost_s.masked_fill_(~mask, 0.0)
    else:
        num = im.size(0)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0)
    # normalized by the number of items. vse_loss[i] is the vse loss for i-th sample
    vse_loss = vse_loss / (num - 1)
    if reduction == "none":
        return vse_loss
    elif reduction == "sum":
        return vse_loss.sum()
    elif reduction == "mean":
        return vse_loss.sum() / num
    else:
        raise KeyError


def soft_margin_loss(pos, neg=None, reduction="none"):
    """Compute the BPR loss."""
    if neg is None:
        x = pos
    else:
        x = pos - neg
    return F.soft_margin_loss(x, torch.ones_like(x), reduction=reduction)
