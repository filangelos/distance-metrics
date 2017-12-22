# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd

from reader import fetch_data
from normaliser import normalise

from metrics import chisquare, correlation, intersection

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from visualize import plot_confusion_matrix
from visualize import plot_kneighbors_graph
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10.0, 10.0]

import argparse


def cross_validate(data):
    """Cross-Validate KNN.

    Parameters
    ----------
    data: dict
        * train: tuple
            - X: features
            - y: labels
        * test: tuple
            - X: features
            - y: labels

    Returns
    -------
    method: std
        Transformation function
    params: dict
        * metric: function | str
            Distance metric function
        * metric_params: dict
            Parameters of `metric` function
    """

    norm_methods = ['none', 'l1', 'l2', 'max',
                    'standard', 'maxabs', 'minmax', 'robust']

    params_grid = [('Intersection', {'metric': intersection}),
                   ('Correlation', {'metric': correlation}),
                   ('Manhattan', {'metric': 'manhattan'}),
                   ('Euclidean', {'metric': 'euclidean'}),
                   ('Chebyshev', {'metric': 'chebyshev'}),
                   ('Chi-Square', {'metric': chisquare})]
    results = {}
    best_params = {}
    best_score = -1

    for method in norm_methods:

        data = normalise(data, method=method)

        X_train, y_train = data['train']
        X_test, y_test = data['test']

        results[method] = {}

        for name, params in params_grid:

            classifier = NearestCentroid(**params)
            acc = cross_val_score(
                classifier, X_train, y_train, cv=3).mean()
            results[method][name] = acc

        best_metric = None
        best_accuracy = -1

        for name, accuracy in results[method].items():
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_metric = name

        best_params_ = {**dict(params_grid)[best_metric]}
        print('')
        print('[%s]' % method, 'Best params:', best_params_)
        print('[%s]' % method, 'Best CV Score:',
              results[method][best_metric])

        best_classifier_ = NearestCentroid(**best_params_)
        best_classifier_.fit(X_train, y_train)

        best_score_ = best_classifier_.score(X_test, y_test)

        print('[%s]' % method, 'Accuracy:', best_score_)

        if best_score_ > best_score:
            print('[%s]' % method, 'New Best:', best_params_)
            best_params = (method, best_params_)
            best_score = best_score_
    print('')
    print('Cross Validation Results:', best_params)
    print('')
    return best_params


def main(CV=False, PLOT=True):
    """Entry Point.

    Parameters
    ----------
    CV: bool
        Cross-validation flag
    PLOT: bool
        Plotting flag
    """
    _data = fetch_data()

    if CV:
        method, params = cross_validate(_data)
    else:
        method = 'l2'
        params = {'metric': chisquare}

    data = normalise(_data, method)

    X_train, y_train = data['train']
    X_test, y_test = data['test']

    classifier = NearestCentroid(**params)
    classifier.fit(X_train, y_train)

    print('ACCURACY: ', classifier.score(X_test, y_test))

    if PLOT:

        y_hat = classifier.predict(X_test)

        cnf_matrix = confusion_matrix(y_test, y_hat)

        plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
                              title='Nearest Centroid\nConfusion Matrix',
                              cmap=plt.cm.Blues)

        plt.savefig('data/out/nc_cnf_matrix.pdf',
                    format='pdf', dpi=300, transparent=True)


if __name__ == '__main__':
    # argument parser instance
    parser = argparse.ArgumentParser()
    # init log level argument
    parser.add_argument('-p', '--plot', action='store_true',
                        help='<optional> Plot Graphs Flag')
    parser.add_argument('-cv', '--cross_validation', action='store_true',
                        help='<optional> Cross-Validation Flag')
    # parse arguments
    argv = parser.parse_args()
    # get flag of plotting
    plot = argv.plot or False
    # get flag of cross validation
    cv = argv.cross_validation or False
    # execute program
    main(CV=cv, PLOT=plot)
