# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          new_fig=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if new_fig:
        plt.figure()
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, size=20)
    plt.yticks(tick_marks, classes, size=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontdict={'size': 40},
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)
    plt.tight_layout()


def plot_kneighbors_graph(matrix, title='Neighbours Graph', cmap='hot', new_fig=True):
    if new_fig:
        plt.figure()
    plt.imshow(matrix.toarray(), interpolation='nearest', cmap='hot')
    plt.title(title, fontsize=20)
    plt.ylabel('$x_{i}^{(test)}$', fontsize=20)
    plt.xlabel('$x_{j}^{(train)}$', fontsize=20)
    plt.tight_layout()
