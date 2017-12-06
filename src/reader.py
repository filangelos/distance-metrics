# scientific computing library
import numpy as np


def fetch_data(fname='wine.data', ratio=0.8):
    """Bootstrapping helper function for fetching data.

    Parameters
    ----------
    fname: str
        Name of the `.csv` input file
    ratio: float
        Split ratio of dataset

    Returns
    -------
    data: dict
        * train: tuple
            - X: features
            - y: labels
        * test: tuple
            - X: features
            - y: labels
    """

    # load `.csv` file
    data = np.loadtxt('data/%s.csv' % fname, delimiter=' ')

    return data
