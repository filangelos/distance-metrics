from reader import fetch_data
from visualize import plot_confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data = fetch_data()


raw_X_train, y_train = data['train']
raw_X_test, y_test = data['test']

scaler = StandardScaler().fit(raw_X_train)

X_train = scaler.transform(raw_X_train)

classifier = KNeighborsClassifier(n_neighbors=14)

classifier.fit(X_train, y_train)

X_test = scaler.transform(raw_X_test)

y_hat = classifier.predict(X_test)
"""
cnf_matrix = confusion_matrix(y_test, y_hat)

plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
                      title='Nearest Neighbor\nConfusion Matrix',
                      cmap=plt.cm.Greens)

A = classifier.steps[-1][1].kneighbors_graph(X_test)

plt.figure()

plt.imshow(A.toarray(), cmap='hot', interpolation='nearest')

# plt.show()
"""
print(classifier.score(X_test, y_test))
