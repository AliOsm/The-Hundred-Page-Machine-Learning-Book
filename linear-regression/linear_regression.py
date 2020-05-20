import numpy as np


class LinearRegression():
    def __init__(self, X, y, n_iter=1000, lr=0.1, normalize=True, use_biases=True):
        assert(X.shape[0] == len(y))

        if normalize:
            X = self.__normalize(X)

        if use_biases:
            X = self.__use_biases(X)

        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.lr = lr

        self.n_samples = len(y)
        self.n_features = X.shape[1]
        self.params = np.ones((X.shape[1], 1))
        self.losses = np.zeros((n_iter, 1))


    def fit(self):
        for i in range(self.n_iter):
            self.params = self.params - self.lr * self.__derivatives()
            self.losses[i] = self.__mse_loss()
            print('Iteration #{} - Loss {}'.format(i + 1, self.losses[i]))
        return self.losses


    def predict(self, X, normalize=True, use_biases=True):      
        if normalize:
            X = self.__normalize(X)

        if use_biases:
            X = self.__use_biases(X)

        return X.dot(self.params)


    def get_params(self):
        return self.params


    def __mse_loss(self):
        y_hat = self.X.dot(self.params)
        return (1 / (2 * self.n_samples)) * np.power(y_hat - self.y, 2).sum()


    def __derivatives(self):
        return self.X.T.dot((self.X.dot(self.params) - self.y)) / self.n_samples


    def __normalize(self, X):
        return (X - X.mean()) / X.std()


    def __use_biases(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
