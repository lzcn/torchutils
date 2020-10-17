import numpy as np
import scipy as sp


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


def make_2D_samples_gauss(n, m, sigma, norm=True):
    """Return n samples drawn from 2D gaussian N(m,sigma)

    Parameters
    ----------
    n : int
        number of samples to make
    m : ndarray, shape (2,)
        mean value of the gaussian distribution
    sigma : ndarray, shape (2, 2)
        covariance matrix of the gaussian distribution
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : ndarray, shape (n, 2)
        n samples drawn from N(m, sigma).
    """

    if np.isscalar(sigma):
        sigma = np.array([sigma,])
    if len(sigma) > 1:
        P = sp.linalg.sqrtm(sigma)
        res = generator.randn(n, 2).dot(P) + m
    else:
        res = generator.randn(n, 2) * np.sqrt(sigma) + m
    return res
