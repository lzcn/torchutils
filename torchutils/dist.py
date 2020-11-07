import numpy as np


def make_1D_gauss(n, mean=0, std=1.0, norm=True):
    """return a 1D histogram for a gaussian distribution

    Parameters
    ----------
    n : int
        number of bins in the histogram
    mean : float
        mean value of the gaussian distribution
    std : float
        standard deviaton of the gaussian distribution

    Returns
    -------
    h : ndarray (n,)
        1D histogram for a gaussian distribution
    """
    x = np.arange(n, dtype=np.float64)
    h = np.exp(-0.5 * ((x - mean) ** 2) / (std ** 2))
    if norm:
        return h / h.sum()
    else:
        return h
