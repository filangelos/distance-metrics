# scientific computing library
import numpy as np


def fetch_data(fname='wine.data', ratio=0.8, standard=True):
    """Bootstrapping helper function for fetching data.

    Parameters
    ----------
    fname: str
        Name of the `.csv` input file
    ratio: float
        Split ratio of dataset
    standard: bool


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
    train_mask = data[:, 0] == 1

    #
    test_mask = data[:, 0] == 2
    #
    X_train = data[train_mask, 2:].astype(float)
    y_train = data[train_mask, 1].astype(int)
    y_train -= 1

    X_test = data[test_mask, 2:].astype(float)
    y_test = data[test_mask, 1].astype(int)
    y_test -= 1

    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
