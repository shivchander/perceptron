#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Perceptron classifier
'''
from perceptron import *
np.random.seed(0)


def gen_syn_data(num_points, num_features, test_size):
    X = np.random.randint(-10, 10, (num_points, num_features))
    y = (X[:, 0] + (2 * X[:, 1]) - 2 > 0).astype(int)
    y[np.where(y == 0)] = -1
    y = y.reshape(num_points, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = gen_syn_data(100, 2, 0.5)

# fig, axs = plt.subplots(4)
# for i, lr in enumerate([0.1, 0.01, 0.001, 0.0001]):
#     model = Perceptron()
#     model.fit(X_train, y_train, 100, 0.001, optimizer='stochastic')
#     axs[i].plot(model.errors)
# plt.show()


model = Perceptron()
model.fit(X_train, y_train, 100, 0.001, optimizer='batch', adaptive_rate_threshold=1.02, early_stopping=True)
plt.plot(model.errors)
plt.show()