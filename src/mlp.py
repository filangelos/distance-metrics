# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd

from reader import fetch_data
from normaliser import normalise

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

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

    """

    norm_methods = ['none', 'l1', 'l2', 'max',
                    'standard', 'maxabs', 'minmax', 'robust']

    results = {}

    best_method = None
    best_score = -1

    for method in norm_methods:

        data = normalise(data, method=method)

        X_train, y_train = data['train']
        X_test, y_test = data['test']

        params = {'hidden_layer_sizes': [
            (25,), (50,), (100,), (10, 10), (10, 25)], 'activation': ['logistic', 'tanh', 'relu']}

        search = GridSearchCV(MLPClassifier(
            learning_rate="adaptive",
            max_iter=5000, solver='adam', random_state=42, alpha=0.01), params)

        search.fit(X_train, y_train)

        print('[%s]' % method, 'Best params:', search.best_params_)

        classifier = search.best_estimator_

        results[method] = classifier.score(X_test, y_test)

        print('[%s]' % method, 'Accuracy:', results[method])

        if results[method] > best_score:
            print('[%s]' % method, 'New Best:', search.best_params_)
            best_params = (method, search.best_params_)
            best_score = results[method]
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
        method = 'robust'
        params = {'activation': 'logistic', 'hidden_layer_sizes': (25,)}

    data = normalise(_data, method)

    X_train, y_train = data['train']
    X_test, y_test = data['test']

    classifier = MLPClassifier(learning_rate="adaptive",
                               max_iter=5000, solver='adam', random_state=42, alpha=0.01, **params)
    classifier.fit(X_train, y_train)

    print('ACCURACY: ', classifier.score(X_test, y_test))

    if PLOT:

        y_hat = classifier.predict(X_test)

        cnf_matrix = confusion_matrix(y_test, y_hat)

        plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
                              title='Multi-Layer-Perceptron\nConfusion Matrix',
                              cmap=plt.cm.Reds)

        plt.savefig('data/out/mlp_cnf_matrix.pdf',
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
