from reader import fetch_data
from visualize import plot_confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = fetch_data()


X_train, y_train = data['train']
X_test, y_test = data['test']

V = np.cov(X_train)

classifier = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis', metric_params={'V': V}, algorithm='brute')

classifier.fit(X_train, y_train)

y_hat = classifier.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_hat)

plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
                          title='Nearest Neighbor - Confusion Matrix',
                          cmap=plt.cm.Greens)


A = classifier.kneighbors_graph(X_test)

plt.figure()

plt.imshow(A.toarray(), cmap='hot', interpolation='nearest')

plt.show()
