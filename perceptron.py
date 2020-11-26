import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, classification_report, accuracy_score
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        self.errors = []

    def threshold_function(self, x):
        fx = np.dot(self.w, x)+self.b
        return 1 if fx > 0 else -1

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.threshold_function(x))

        return y_pred

    def fit(self, X, y, epochs=50, alpha=0.1, optimizer='stochastic', plot_error=False, plot_decision=False,
            decaying_rate=None, adaptive_rate_threshold=None, early_stopping=False):
        self.w = np.random.randn(X.shape[1]).reshape(1, X.shape[1])
        self.b = 0
        weights_history = []
        bias_history = []

        for i in range(epochs):
            sum_dw = np.zeros(X.shape[1]).reshape(1, X.shape[1])
            sum_db = 0

            for xi, yi in zip(X, y):
                y_pred = self.threshold_function(xi)
                dw = alpha * (yi - y_pred) * xi
                db = alpha * (yi - y_pred) * 1
                sum_dw = sum_dw + dw
                sum_db = sum_db + db
                if optimizer == 'stochastic':
                    self.w = self.w + dw
                    self.b = self.b + db
                    weights_history.append(self.w)
                    bias_history.append(self.b)

            if optimizer == 'batch':
                self.w = self.w + sum_dw
                self.b = self.b + sum_db
                weights_history.append(self.w)
                bias_history.append(self.b)

            error = mean_squared_error(y, self.predict(X))
            # print(error)
            self.errors.append(error)

            if early_stopping:
                if error <= 0.01:
                    print('Stopping Training')
                    break

            if decaying_rate:
                alpha = alpha * decaying_rate
                # print('a: ', alpha)

            if adaptive_rate_threshold:
                if i > 0:
                    if self.errors[-1] - self.errors[-2] > adaptive_rate_threshold:
                        self.w = weights_history[-2]
                        self.b = bias_history[-2]
                        alpha = alpha * 0.9
                        # print('a: ', alpha)
                    else:
                        alpha = alpha * 1.1
                        # print('a: ', alpha)

            if plot_decision:
                if i+1 in [5, 10, 50, 100]:
                    self.decision_boundary(X, y, title='{} epoch: {}'.format(optimizer, i+1))

        if plot_error:
            plt.plot(self.errors)
            plt.show()

    def decision_boundary(self, X, y, title):
        min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
        min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
        # define the x and y scale
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)
        # create all of the lines and rows of the grid
        xx, yy = np.meshgrid(x1grid, x2grid)
        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        # horizontal stack vectors to create x1,x2 input for the model
        grid = np.hstack((r1, r2))
        yhat_perceptron = np.array(self.predict(grid))

        zz_perceptron = yhat_perceptron.reshape(xx.shape)

        plt.contourf(xx, yy, zz_perceptron, cmap='Paired')
        for class_value in [-1, 1]:
            # get row indexes for samples with this class
            row_ix = np.where(y == class_value)
            # create scatter of these samples
            plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        plt.title('Perceptron Decision Boundary - {}'.format(title))
        plt.xlabel('x1')
        plt.ylabel('x2')
        # plt.savefig('figs/{}.pdf'.format(title))
        plt.show()
        plt.clf()
