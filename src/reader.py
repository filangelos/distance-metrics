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

    # split data [train, test]
    train_mask = data[:, 0] == 1 # rows mask

    #
    test_mask = data[:, 0] == 2

    #
    X_train = data[train_mask,2:]
    y_train = data[train_mask,1].astype(int)

    X_test = data[test_mask,2:]
    y_test = data[test_mask,1].astype(int)

    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
