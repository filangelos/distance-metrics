# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from reader import fetch_data
from normaliser import normalise

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns_blue, sns_green, sns_red, _, _, _ = sns.color_palette("muted")
sns.set_style("ticks")
plt.rcParams['figure.figsize'] = [6.0, 12.0]

fig, axes = plt.subplots(nrows=4, ncols=2)

tuples = [(axes[0, 0], 'none', 'Raw'),
          (axes[0, 1], 'l2', 'L2 Normalised'),
          (axes[1, 0], 'l1', 'L1 Normalised'),
          (axes[1, 1], 'max', '$L_{\infty}$ Normalised'),
          (axes[2, 0], 'standard', 'Standardardised'),
          (axes[2, 1], 'maxabs', 'Maximum Absolute Value Scaled'),
          (axes[3, 0], 'minmax', 'Minimum to Maximum Values Scaled'),
          (axes[3, 1], 'robust', 'IQR and Median Scaled')]

for ax, method, title in tuples:

    data = normalise(data=fetch_data(), method=method)

    X_train, y_train = data['train']
    X_test, y_test = data['test']

    pca = PCA(n_components=2)

    W_train = pca.fit_transform(X_train)
    W_test = pca.transform(X_test)

    _drawn = [False, False, False]
    col = [sns_blue, sns_green, sns_red]

    for w, y in zip(W_train, y_train):
        if not _drawn[y - 1]:
            ax.scatter(w[0], w[1], c=col[y - 1],
                       label='%s' % (y + 1))
            _drawn[y - 1] = True
        else:
            ax.scatter(w[0], w[1], c=col[y - 1])
    ax.legend(frameon=True)
    # ax.set_xlabel('$w_{1}$')
    # ax.set_ylabel('$w_{2}$')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)

plt.savefig('data/out/standard_comparison.pdf',
            format='pdf', dpi=300, transparent=True)
