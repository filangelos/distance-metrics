import numpy as np

from scipy.spatial.distance import correlation as _correlation
np.seterr(divide='ignore', invalid='ignore')


def minkowski(x1, x2, power):
    """Minkowski Distance Metric

    Parameters
    ----------
    x1: numpy.ndarray
        Vector one
    x2: numpy.ndarray
        Vector two
    power: int
        L_{power} norm order

    Returns
    -------
    distance: float
        Minkowski distance between `x1` and `x2`
    """
    return np.linalg.norm(x1 - x2, power)


def cosine(x1, x2):
    """Cosine Distance Metric

    Parameters
    ----------
    x1: numpy.ndarray
        Vector one
    x2: numpy.ndarray
        Vector two

    Returns
    -------
    distance: float
        Cosine distance between `x1` and `x2`
    """
    return np.dot(x1.T, x2) / (np.linalg.norm(x1, 2) * np.linalg.norm(x2, 2))


def chisquare(x1, x2):
    """Chi-Square Distance Metric

    Parameters
    ----------
    x1: numpy.ndarray
        Vector one
    x2: numpy.ndarray
        Vector two

    Returns
    -------
    distance: float
        Chi-Square distance between `x1` and `x2`
    """
    return np.sum((x1 - x2)**2 / (x1 + x2))


def correlation(x1, x2):
    """Correlation Distance

    Parameters
    ----------
    x1: numpy.ndarray
        Vector one
    x2: numpy.ndarray
        Vector two

    Returns
    -------
    distance: float
        Correlation distance between `x1` and `x2`
    """
    return _correlation(x1, x2)


def intersection(x1, x2):
    """Histogram Intersection

    Parameters
    ----------
    x1: numpy.ndarray
        Vector one
    x2: numpy.ndarray
        Vector two

    Returns
    -------
    distance: float
        Histogram intersection between `x1` and `x2`
    """
    assert(len(x1) == len(x2))
    minsum = 0
    for i in range(len(x1)):
        minsum += min(x1[i], x2[i])
    return float(minsum) / min(sum(x1), sum(x2))
