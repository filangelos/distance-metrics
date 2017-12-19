# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from reader import fetch_data
from visualize import plot_confusion_matrix
from visualize import plot_kneighbors_graph

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20.0, 15.0]

from metrics import minkowski, cosine, chisquare
from sklearn.covariance import EmpiricalCovariance, MinCovDet

data = fetch_data(ratio=0.8, standard=True)

X_train, y_train = data['train']
X_test, y_test = data['test']

param_grid = [{'metric': [minkowski], 'metric_params': [{'power': 1}, {'power': 2}, {'power': np.inf}]},
              {'metric': ['mahalanobis'], 'metric_params': [{'V': EmpiricalCovariance().fit(X_train).covariance_}, {'V': MinCovDet().fit(X_train).covariance_}, {'V': np.cov(X_train.T)}]}]

search = GridSearchCV(KNeighborsClassifier(
    n_neighbors=4), param_grid)

search.fit(X_train, y_train)

print('Best params:', search.best_params_)

classifier = search.best_estimator_

y_hat = classifier.predict(X_test)

print('Accuracy:', classifier.score(X_test, y_test))

cnf_matrix = confusion_matrix(y_test, y_hat)

plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
                      title='Nearest Neighbor\nConfusion Matrix',
                      cmap=plt.cm.Greens)

plt.savefig('data/out/knn_cnf_matrix.pdf', format='pdf', dpi=300)

neighbors_matrix = classifier.kneighbors_graph(X_test)

plot_kneighbors_graph(neighbors_matrix)

plt.savefig('data/out/knn_neighbours.pdf', format='pdf', dpi=300)
