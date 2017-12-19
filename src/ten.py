import tensorflow as tf
import matplotlib.pyplot as plt
from reader import fetch_data

import numpy as np


def to_one_hot(y):
    max_y = np.max(y)
    x = []
    for yi in y:
        tmp = np.zeros(max_y, dtype=int)
        tmp[-yi] = 1
        x.append(tmp)
    x = np.array(x)
    return x


data = fetch_data()

X_train, y_train = data['train']
X_test, y_test = data['test']

#y_train = to_one_hot(y_train + 1)
#y_test = to_one_hot(y_test + 1)

# init model
tf_X = tf.placeholder(tf.float32, [None, X_train.shape[1]], name='X')
tf_y = tf.placeholder(tf.int32, [None], name='y')

N_UNITS = 100
ACTIVATION = tf.nn.sigmoid
LR = 0.01
EPOCHS = 5000

# hidden layer
l1 = tf.layers.dense(tf_X, N_UNITS, activation=ACTIVATION)
output = tf.layers.dense(l1, 3)

# fit model
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
train_op = tf.train.MomentumOptimizer(LR, momentum=5).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.squeeze(
    tf_y), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

error = []

for step in range(EPOCHS):
    _, l, y_hat, acc = sess.run([train_op, loss, output, accuracy], {
        tf_X: X_train, tf_y: y_train})
    plt.plot(step, l, 'ro', label='Training Error')
    if step % 100 == 0:
        print('Step %d: Accuracy %.3f' % (step, acc))

y_hat = sess.run(output, {tf_X: X_test, tf_y: y_test})

print(y_hat)

acc = np.sum(y_hat == y_test) / y_test.shape[0]

print(acc)

plt.show()
