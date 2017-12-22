# scientific computing library
import numpy as np


def fetch_data(fname='wine.data'):
    """Bootstrapping helper function for fetching data.

    Parameters
    ----------
    fname: str
        Name of the `.csv` input file

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
    test_mask = data[:, 0] == 2

    # train
    X_train = data[train_mask, 2:].astype(float)
    y_train = data[train_mask, 1].astype(int)
    y_train -= 1

    # test
    X_test = data[test_mask, 2:].astype(float)
    y_test = data[test_mask, 1].astype(int)
    y_test -= 1

    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
