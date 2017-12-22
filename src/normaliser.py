# scientific computing library
import numpy as np
# normalisers
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def normalise(data, method='robust'):
    """Normalise `data` with `method`.

    Parameters
    ----------
    data: dict
        * train: tuple
            - X: features
            - y: labels
        * test: tuple
            - X: features
            - y: labels
    method: str
        Rescale (and center) data (per feature) by:
        * l2: unit L2 norm
        * l1: unit L1 norm
        * max: unit L{inf} norm
        * standard: standardise N(0, 1) each feature
        * maxabs: maximum absolute value
        * minmax: minimum and maximum values
        * robust: robust to outliers (IQR and median)
        * none: identity block

    Returns
    -------
    rescaled_data: dict
        * train: tuple
            - X: features
            - y: labels
        * test: tuple
            - X: features
            - y: labels
    """
    if method == 'none':
        return data

    X_train, y_train = data['train']
    X_test, y_test = data['test']

    if method == 'l2':
        trans = Normalizer('l2')
    elif method == 'l1':
        trans = Normalizer('l1')
    elif method == 'max':
        trans = Normalizer('max')
    elif method == 'standard':
        trans = StandardScaler()
    elif method == 'maxabs':
        trans = MaxAbsScaler()
    elif method == 'minmax':
        trans = MinMaxScaler()
    elif method == 'robust':
        trans = RobustScaler()
    else:
        raise ValueError('Unrecognised method=%s' % method)

    X_train = trans.fit_transform(X_train)
    X_test = trans.transform(X_test)

    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
