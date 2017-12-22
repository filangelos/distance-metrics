# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd

from reader import fetch_data
from normaliser import normalise

from metrics import chisquare, correlation, intersection

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from visualize import plot_confusion_matrix
from visualize import plot_kneighbors_graph
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10.0, 10.0]

import argparse


def cross_validate(data, K_max):
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
    K_max: int
        Maximum number of neighbours

    Returns
    -------
    method: std
        Transformation function
    params: dict
        * n_neighbors: int
            Number of neighbours
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

        results[method] = pd.DataFrame(index=np.arange(1, K_max + 1),
                                       columns=list(zip(*params_grid))[0],
                                       dtype=float)

        for k in results[method].index:

            for name, params in params_grid:

                classifier = KNeighborsClassifier(n_neighbors=k, **params)
                acc = cross_val_score(
                    classifier, X_train, y_train, cv=3).mean()
                results[method].loc[k, name] = acc

        idxmax = np.unravel_index(
            np.argmax(results[method].values), results[method].values.shape)

        best_params_ = {'n_neighbors': results[method].index[idxmax[0]], **dict(
            params_grid)[results[method].columns[idxmax[1]]]}
        print('')
        print('[%s]' % method, 'Best params:', best_params_)
        print('[%s]' % method, 'Best CV Score:',
              results[method].values[idxmax])

        best_classifier_ = KNeighborsClassifier(**best_params_)
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
        method, params = cross_validate(_data, 10)
    else:
        method = 'l2'
        params = {'n_neighbors': 1, 'metric': chisquare}

    data = normalise(_data, method)

    X_train, y_train = data['train']
    X_test, y_test = data['test']

    classifier = KNeighborsClassifier(**params)
    classifier.fit(X_train, y_train)

    print('ACCURACY: ', classifier.score(X_test, y_test))

    if PLOT:

        y_hat = classifier.predict(X_test)

        cnf_matrix = confusion_matrix(y_test, y_hat)

        plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
                              title='K-Nearest-Neighbours\nConfusion Matrix',
                              cmap=plt.cm.Greens)

        plt.savefig('data/out/knn_cnf_matrix.pdf',
                    format='pdf', dpi=300, transparent=True)

        neighbors_matrix = classifier.kneighbors_graph(X_test)

        plot_kneighbors_graph(neighbors_matrix, title='Neighbours Graph')

        plt.savefig('data/out/knn_neighbours.pdf',
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
