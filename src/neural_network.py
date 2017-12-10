from reader import fetch_data
from visualize import plot_confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = fetch_data()

X_train, y_train = data['train']
X_test, y_test = data['test']

params = {'hidden_layer_sizes': [
    (100,), (100, 100), (10, 10), (1000, 1000), ()], 'activation': ['logistic', 'tanh']}

search = GridSearchCV(MLPClassifier(
    learning_rate="adaptive", activation="logistic", max_iter=1000), params)

search.fit(X_train, y_train)

print(search.best_params_)

classifier = search.best_estimator_

#classifier.fit(X_train, y_train)

y_hat = classifier.predict(X_test)

acc = classifier.score(X_test, y_test)

print(acc)

# cnf_matrix = confusion_matrix(y_test, y_hat)

# plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
#                       title='Nearest Neighbor - Confusion Matrix',
#                       cmap=plt.cm.Greys)

# plt.show()
