from reader import fetch_data
from visualize import plot_confusion_matrix
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = fetch_data()

X_train, y_train = data['train']
X_test, y_test = data['test']

cluster = KMeans(n_clusters=3, random_state=13)

cluster.fit(X_train)

# <cluster center> -> <label class> (kmeans)
# X -> y (knn)
labels_map = [(cluster.cluster_centers_[np.argmax(np.bincount(cluster.labels_[y_train == l]))], l)
              for l in set(cluster.labels_)]

X, y = zip(*labels_map)

V = np.cov(X)

classifier = KNeighborsClassifier(
    n_neighbors=1, metric='mahalanobis', metric_params={'V': V}, algorithm='brute')

classifier.fit(X, y)

accuracy = classifier.score(X_test, y_test)
