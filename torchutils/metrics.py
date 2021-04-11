import logging
from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)


def ndcg(y_true: List, y_score: List, wtype: str = "max") -> List:
    """Compute the NDCG score

    Args:
        y_true (List): shape: math:`N`. true score.
        y_score (List): shape: math:`N`. Predicted scores.
        wtype (str, optional): tpye for discounts. Defaults to "max".

    Returns:
        List: NDCG@m

    References:

        - [1] Hu Y, Yi X, Davis L S. Collaborative fashion recommendation:
           A functional tensor factorization approach[C]
           Proceedings of the 23rd ACM international conference on Multimedia.
           ACM, 2015: 129-138.
        - [2] Lee C P, Lin C J. Large-scale Linear RankSVM[J].
           Neural computation, 2014, 26(4): 781-817.
    """
    y_score = y_score.reshape(-1)
    y_true = y_true.reshape(-1)
    order = np.argsort(-y_score)
    p_label = np.take(y_true, order)
    i_label = np.sort(y_true)[::-1]
    p_gain = 2 ** p_label - 1
    i_gain = 2 ** i_label - 1
    if wtype.lower() == "max":
        discounts = np.log2(np.maximum(np.arange(len(y_true)) + 1, 2.0))
    else:
        discounts = np.log2(np.arange(len(y_true)) + 2)
    dcg_score = (p_gain / discounts).cumsum()
    idcg_score = (i_gain / discounts).cumsum()
    return dcg_score / idcg_score


def to_canonical(pos: list, neg: list) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return the canonical representation for computing AUC.

    Args:
        pos (list): positive scores
        neg (list): negative scores

    Returns:
        Tuple[np.ndarray, np.ndarray]:

            1. `y_label`: concatenated positive and negative labels
            2. `y_score`: concatenated positive and negative scores

    """
    pos, neg = np.array(pos), np.array(neg)
    y_label = np.array([1] * len(pos) + [0] * len(neg))
    y_score = np.hstack((pos.flatten(), neg.flatten()))
    return (y_label, y_score)


def auc_score(pos, neg):
    r"""Compute auc.

    Args:
        pos (list) : score for positive outfit
        neg (list) : score for negative outfit
    Return:
        auc score
    """
    y_true, y_score = to_canonical(pos, neg)
    return roc_auc_score(y_true, y_score)


def ndcg_score(pos, neg):
    r"""Compute mean ndcg score.

    Args:
        pos (list) : score for positive outfit
        neg (list) : score for negative outfit
    Return
        ndcg (float): ndcg score
    """
    y_label, y_score = to_canonical(pos, neg)
    return ndcg(y_label, y_score).mean()


def pair_accuracy(pos: list, neg: list) -> float:
    r"""Compute pairwise accuracy.

    Give positive and negative scores :math:`x,y\in\mathbb{R}^d`

    .. math::

        \text{accuracy}=\frac{\sum_{i,j} \mathbb{I}(x_i > y_j)}{d^2}

    Args:
        pos (list): positive scores
        neg (list): negative scores
    Return:
        float: pairwise accuracy
    """
    pos = np.array(pos)
    neg = np.array(neg)
    diff = pos.reshape(-1, 1) - neg.reshape(1, -1)
    return (diff > 0).sum() / diff.size


def pair_rank_loss(pos, neg):
    r"""Compute pairwise rank loss.

    Give positive and negative scores :math:`x,y\in\mathbb{R}^d`

    .. math::
        l=\frac{1}{d^2}\sum_{i,j} \log\left(1.0+\exp(y_j-x_i))\right)

    Args:
        pos (list) : score for positive outfit
        neg (list) : score for negative outfit
    Return:
        float: pair-wise loss
    """
    pos = np.array(pos)
    neg = np.array(neg)
    diff = pos.reshape(-1, 1) - neg.reshape(1, -1)
    return np.log(1.0 + np.exp(-diff)).sum() / diff.size


def margin_loss(pos, neg, margin=0.1):
    r"""Compute margin loss

    Give positive and negative scores :math:`x,y\in\mathbb{R}^d`:

    .. math::
        l=\frac{1}{d}\sum_i\left(\max(1-m-x_i,0) + \max(y_i-m,0)\right)
    Args:
        pos (list): score for positive outfit
        neg (list): score for negative outfit

    Returns:
        float: margin loss
    """
    pos = np.maximum(1.0 - margin - np.array(pos), 0) ** 2
    neg = np.maximum(np.array(neg) - margin, 0) ** 2
    return (pos.sum() + neg.sum()) / (pos.size + neg.size)
