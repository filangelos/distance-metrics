import numpy as np


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
