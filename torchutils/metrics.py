import logging
import threading

import numpy as np
import sklearn.metrics

_POSILABEL = 1
_NEGALABEL = 0

LOGGER = logging.getLogger(__name__)


def _canonical(posi, nega):
    """Return the canonical representation.

    Parameters
    ----------
    posi: positive scores
    nege: negative scores

    Return
    ------
    y_true: true label 0 for negative sample and 1 for positive
    y_score: predicted score for corresponding samples

    """
    posi, nega = np.array(posi), np.array(nega)
    y_true = np.array([_POSILABEL] * len(posi) + [_NEGALABEL] * len(nega))
    y_score = np.hstack((posi.flatten(), nega.flatten()))
    return (y_true, y_score)


def _precision(posi, nega):
    y_score = np.hstack((posi.flatten(), nega.flatten()))
    num = len(posi)
    rank = np.argsort(-y_score)
    adopted = set(range(num))
    rec = set()
    precision = []
    for i in range(num):
        rec.add(rank[i])
        p = 1.0 * len(adopted & rec) / (i + 1)
        precision.append(p)
    return np.array(precision)


def Precision(posi, nega):
    """Compute precision for all user."""
    num = max([len(p) for p in posi])
    precision = np.zeros(num)
    count = np.zeros(num)
    for p, n in zip(np.array(posi), np.array(nega)):
        prec = _precision(p, n)
        precision[0 : len(prec)] += prec
        count[0 : len(prec)] += 1
    return precision / count


def ROC(posi, nega):
    """Compute ROC and AUC for all user.

    Parameters
    ----------
    posi: Scores for positive outfits for each user.
    nega: Socres for negative outfits for each user.

    Returns
    -------
    roc: A tuple of (fpr, tpr, thresholds), see sklearn.metrics.roc_curve
    auc: AUC score.

    """
    assert len(posi) == len(nega)
    num = len(posi)
    mean_auc = 0.0
    aucs = []
    for p, n in zip(posi, nega):
        y_true, y_score = _canonical(p, n)
        auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        aucs.append(auc)
        mean_auc += auc
    mean_auc /= num
    return (aucs, mean_auc)


def calc_AUC(posi, nega):
    _, avg_auc = ROC(posi, nega)
    return avg_auc


def calc_NDCG(posi, nega):
    mean_ndcg, _ = NDCG(posi, nega)
    return mean_ndcg.mean()


def NDCG(posi, nega, wtype="max"):
    """Mean Normalize Discounted cumulative gain (NDCG).

    Parameters
    ----------
    posi: positive scores for each user.
    nega: negative scores for each user.
    wtype: type for discounts

    Returns
    -------
    mean_ndcg : array, shape = [num_users]
        mean ndcg for each user (averaged among all rank)
    avg_ndcg : array, shape = [max(n_samples)], averaged ndcg at each
        position (averaged among all users for given rank)

    """
    assert len(posi) == len(nega)
    u_labels, u_scores = [], []
    for p, n in zip(posi, nega):
        label, score = _canonical(p, n)
        u_labels.append(label)
        u_scores.append(score)
    return mean_ndcg_score(u_scores, u_labels, wtype)


def ndcg_score(y_score, y_label, wtype="max"):
    """Normalize Discounted cumulative gain (NDCG).

    Parameters
    ----------
    y_score : array, shape = [n_samples]
        Predicted scores.
    y_label : array, shape = [n_samples]
        Ground truth label (binary).
    wtype : 'log' or 'max'
        type for discounts
    Returns
    -------
    score : ndcg@m
    References
    ----------
    .. [1] Hu Y, Yi X, Davis L S. Collaborative fashion recommendation:
           A functional tensor factorization approach[C]
           Proceedings of the 23rd ACM international conference on Multimedia.
           ACM, 2015: 129-138.
       [2] Lee C P, Lin C J. Large-scale Linear RankSVM[J].
           Neural computation, 2014, 26(4): 781-817.

    """
    order = np.argsort(-y_score)
    p_label = np.take(y_label, order)
    i_label = np.sort(y_label)[::-1]
    p_gain = 2 ** p_label - 1
    i_gain = 2 ** i_label - 1
    if wtype.lower() == "max":
        discounts = np.log2(np.maximum(np.arange(len(y_label)) + 1, 2.0))
    else:
        discounts = np.log2(np.arange(len(y_label)) + 2)
    dcg_score = (p_gain / discounts).cumsum()
    idcg_score = (i_gain / discounts).cumsum()
    return dcg_score / idcg_score


def mean_ndcg_score(u_scores, u_labels, wtype="max"):
    """Mean Normalize Discounted cumulative gain (NDCG) for all users.

    Parameters
    ----------
    u_score : array of arrays, shape = [num_users]
        Each array is the predicted scores, shape = [n_samples[u]]
    u_label : array of arrays, shape = [num_users]
        Each array is the ground truth label, shape = [n_samples[u]]
    wtype : 'log' or 'max'
        type for discounts
    Returns
    -------
    mean_ndcg : array, shape = [num_users]
        mean ndcg for each user (averaged among all rank)
    avg_ndcg : array, shape = [max(n_samples)], averaged ndcg at each
        position (averaged among all users for given rank)

    """
    num_users = len(u_scores)
    n_samples = [len(scores) for scores in u_scores]
    max_sample = max(n_samples)
    count = np.zeros(max_sample)
    mean_ndcg = np.zeros(num_users)
    avg_ndcg = np.zeros(max_sample)
    for u in range(num_users):
        ndcg = ndcg_score(u_scores[u], u_labels[u], wtype)
        avg_ndcg[: n_samples[u]] += ndcg
        count[: n_samples[u]] += 1
        mean_ndcg[u] = ndcg.mean()
    return mean_ndcg, avg_ndcg / count


def AA(A, pos, neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, pos, neg)


def CN(A, pos, neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, pos, neg)


def CalcAUC(sim, pos, neg):
    """Calculate AUC.
    Score of link (i,j) assigned as the sim(i,j)

    Parameters
    ----------
    sim: similarity matrix
    """
    # Compute the score for positive links and negative links
    pos_scores = np.asarray(sim[pos[0], pos[1]]).squeeze()
    neg_scores = np.asarray(sim[neg[0], neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, scores, pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc


class MetricThreading(threading.Thread):
    def __init__(self, args=(), kwargs=None):
        from queue import Queue

        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.queue = Queue()
        self.deamon = True
        self._pos = []
        self._neg = []

    def reset(self):
        self._pos = []
        self._neg = []

    def put(self, data):
        # if not self.is_alive():
        #     self.start()
        self.queue.put(data)

    def process(self, data):
        with threading.Lock():
            pos, neg = data
            for p, n in zip(pos, neg):
                self._pos.append(p)
                self._neg.append(n)

    def run(self):
        LOGGER.info(
            "Starting %s (%s)", threading.currentThread().getName(), self.__class__.__name__,
        )
        while True:
            data = self.queue.get()
            if data is None:  # If you send `None`, the thread will exit.
                return
            self.process(data)

    def rank(self):
        auc = calc_AUC([self._pos], [self._neg])
        ndcg = calc_NDCG([self._pos], [self._neg])
        return auc, ndcg


class UserMetricThreading(threading.Thread):
    def __init__(self, num_users, args=(), kwargs=None):
        from queue import Queue

        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.queue = Queue()
        self.deamon = True
        self.num_users = num_users
        self._pos = [[] for _ in range(self.num_users)]
        self._neg = [[] for _ in range(self.num_users)]

    def reset(self):
        self._pos = [[] for _ in range(self.num_users)]
        self._neg = [[] for _ in range(self.num_users)]

    def put(self, data):
        # if not self.is_alive():
        #     self.start()
        self.queue.put(data)

    def process(self, data):
        with threading.Lock():
            uidx, pos, neg = data
            for u, p, n in zip(uidx, pos, neg):
                self._pos[u].append(p)
                self._neg[u].append(n)

    def run(self):
        LOGGER.info(
            "Starting %s (%s)", threading.currentThread().getName(), self.__class__.__name__,
        )
        while True:
            data = self.queue.get()
            if data is None:  # If you send `None`, the thread will exit.
                return
            self.process(data)

    def rank(self):
        auc = calc_AUC(self._pos, self._neg)
        ndcg = calc_NDCG(self._pos, self._neg)
        return auc, ndcg


class RankMetric(object):
    """Rank matric for computing mean auc and mean averaged ndch.

    Parameters
    ----------
    num_users : int
        number of users

    groups : list of str
       different kinds of metric will be tracked.
    """

    def __init__(self, num_users, groups: list):
        if isinstance(num_users, int):
            num_users = [num_users] * len(groups)
        else:
            assert len(num_users) == len(groups)
        self._rank_metrics = dict()
        for num, name in zip(num_users, groups):
            if num == 1:
                self._rank_metrics[name] = MetricThreading()
            else:
                self._rank_metrics[name] = UserMetricThreading(num)

    def reset(self):
        for metric in self._rank_metrics.values():
            metric.reset()

    def rank(self):
        results = dict()
        for name, metric in self._rank_metrics.items():
            auc, ndcg = metric.rank()
            results[name + "_auc"] = auc
            results[name + "_ndcg"] = ndcg
        return results

    def stop(self):
        for name, metric in self._rank_metrics.items():
            if metric and metric.is_alive():
                LOGGER.info("Stoping %s (%s)", name, metric.__class__.__name__)
                metric.put(None)
                metric.join()

    def start(self):
        for _, metric in self._rank_metrics.items():
            if not metric.is_alive():
                metric.start()

    def is_alive(self):
        alive = [v.is_alive() for v in self._rank_metrics.values()]
        return all(alive)

    def put(self, group_data):
        for name, data in group_data.items():
            self._rank_metrics[name].put(data)
