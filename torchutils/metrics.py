import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)


def ndcg(y_true: List, y_score: List, wtype: str = "max") -> List:
    """Compute the NDCG score

    Args:
        y_true (List): shape: math:`N`. true score.
        y_score (List): shape: math:`N`. Predicted scores.
        wtype (str, optional): type for discounts. Defaults to "max".

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
    p_gain = 2**p_label - 1
    i_gain = 2**i_label - 1
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
            1. ``y_label``: concatenated positive and negative labels
            2. ``y_score``: concatenated positive and negative scores

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


class Metric:
    def __init__(self, input_transform=lambda x: x) -> None:
        self._input_transform = input_transform
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._y_true = []
        self._y_pred = []

    def update(self, input: Any) -> None:
        y_pred, y_true = self._input_transform(input)
        self._y_pred.extend(y_pred)
        self._y_true.extend(y_true)

    def compute(self):
        return self(self._y_pred, self._y_true)

    def save(self, file):
        results = self.compute()
        data = {"y_pred": self._y_pred, "y_true": self._y_true}
        data.update(**results)
        with open(file, "w") as f:
            json.dump(data, f)

    def __call__(self, y_pred, y_true) -> Any:
        raise NotImplementedError


class ROC_AUC_Score(Metric):
    def __call__(self, y_pred, y_true) -> Any:
        return roc_auc_score(y_true, y_pred)


class NDCG_Score(Metric):
    def __call__(self, y_pred, y_true) -> Any:
        return ndcg(y_true, y_pred).mean()


class SoftMarginLoss(Metric):
    r"""Metric for a two-class classification logistic loss between input :math:`x` and target :math:`y`.

    The label of postivie class should be 1.

    .. math::

        \ell=\sum_{i} \frac{\log\left(1+\exp(-\mathbb{1}_{(y_i=1)}x_i + \mathbb{1}_{(y_i\neq1)}x_i))\right)}{n}


    """

    def __call__(self, y_pred, y_true) -> Any:
        if len(y_pred) == 0:
            raise RuntimeError("Metric must have at least one example before it can be computed.")
        label_set = np.unique(y_true)
        num_classes = len(label_set)
        assert num_classes == 2, "SoftMarginLoss only supports binary classification"
        assert 1 in label_set, "The label of positive class must be 1"
        y = np.where(y_true == 1, 1, -1)
        x = np.array(y_pred)
        return np.log(1.0 + np.exp(-y * x)).sum() / x.size


class BPRLoss(Metric):
    r"""BPR metric for a two-class classification logistic loss between input :math:`x`
    and target :math:`y`.

    The label of postivie class should be :math:`1`. The :class:`BPRLoss` is similar to
    :class:`SoftMarginLoss`. We first split the score :math:`x` into two parts
    :math:`x^+\in\mathbb{R}^{n^+},x^-\in\mathbb{R}^{n^-}` according to the label
    :math:`y`. And the loss is defined as:

    .. math::

        \ell=\sum_{i,j} \frac{\log\left(1+\exp(-(x^+_i - x_j^-))\right)}{n^+n^-}

    If you want to compute the BPR loss of a list of tuples :math:`(x^+_i, x^-_i)`, use
    the :class:`SoftMarginLoss` and update the results with :math:`(x^+_i - x^-_i, 1)`

    """

    def __call__(self, y_pred, y_true) -> Any:
        label_set = np.unique(y_true)
        num_classes = len(label_set)
        assert num_classes == 2, f"BPRLoss only supports binary classification, but get {num_classes}"
        assert 1 in label_set, "The label of positive class must be 1"
        score = np.array(y_pred).flatten()
        label = np.array(y_true).flatten()
        pos_idx = np.where(label == 1)
        neg_idx = np.where(label != 1)
        pos_score = score[pos_idx]
        neg_score = score[neg_idx]
        if len(pos_score) == len(neg_score):
            diff = pos_score - neg_score
        else:
            diff = pos_score.reshape(-1, 1) - neg_score.reshape(1, -1)
            diff = diff.flatten()
        return np.log(1.0 + np.exp(-diff)).sum() / diff.size


class MetricList(Metric):
    def __init__(self, metric_list, input_transform=lambda x: x) -> None:
        super().__init__(input_transform=input_transform)
        self.metric_list = metric_list

    def __call__(self, y_pred, y_true) -> Any:
        results = []
        for metric in self.metric_list:
            results.append(metric(y_pred, y_true))
        return results


class MetricDcit(Metric):
    def __init__(self, metric_dict, input_transform=lambda x: x) -> None:
        super().__init__(input_transform=input_transform)
        self.metric_dict = metric_dict

    def __call__(self, y_pred, y_true) -> Any:
        results = dict()
        for key, metric in self.metric_dict.items():
            results[key] = metric(y_pred, y_true)
        return results


class UserMetricDict(Metric):
    def __init__(self, metric_dict, input_transform=lambda x: x):
        self._y_pred = defaultdict(list)
        self._y_true = defaultdict(list)
        self._output_transform = input_transform
        self.metric_dict = metric_dict

    def reset(self):
        self._y_pred = defaultdict(list)
        self._y_true = defaultdict(list)

    def update(self, *output: List[list]):
        y_pred, y_true, uidx = self._output_transform(*output)
        assert (
            len(y_pred) == len(y_true) == len(uidx)
        ), f"The length of y_pred ({len(y_pred)}), y_true ({len(y_true)}), and uidx ({len(uidx)}) must be the same"
        for x, y, u in zip(y_pred, y_true, uidx):
            self._y_pred[u].append(x)
            self._y_true[u].append(y)

    def __call__(self, y_pred, y_true) -> Dict:
        metrics = defaultdict(list)
        uidx = y_pred.keys()
        for u in uidx:
            for key, metric in self.metric_dict.items():
                metrics[key].append(metric(np.array(y_pred[u]), np.array(y_true[u])))
        return {k: np.sum(v) / len(v) for k, v in metrics.items()}
