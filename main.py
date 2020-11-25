#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Perceptron classifier
'''
from test import *
np.random.seed(0)


def gen_syn_data(num_points, num_features, test_size):
    X = np.random.randint(-10, 10, (num_points, num_features))
    y = (X[:, 0] + (2 * X[:, 1]) - 2 > 0).astype(int)
    y[np.where(y == 0)] = -1
    y = y.reshape(num_points, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = gen_syn_data(100, 2, 0.5)
model = Perceptron()
model.fit(X_train, y_train, 100, 0.001, optimizer='batch', plot_decision=True)
y_preds = model.predict(X_test)
print(classification_report(y_test, y_preds))
