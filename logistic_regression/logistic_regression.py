import numpy as np


class LogisticRegression():
    def __init__(self, X, y, n_iter=1000, lr=0.01, normalize=True, use_biases=True):
        assert(X.shape[0] == len(y))

        if normalize:
            X = self.__normalize(X)

        if use_biases:
            X = self.__use_biases(X)

        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.lr = lr

        self.n_features = X.shape[1]
        self.params = np.zeros((X.shape[1], y[0].shape[0]))
        self.losses = np.zeros((n_iter, 1))


    def fit(self):
        for i in range(self.n_iter):
            self.params -= self.lr * self.__gradients()
            self.losses[i] = self._cross_entropy_loss()
            print('Iteration #{} - Loss {}'.format(i + 1, self.losses[i]))
        return self.losses


    def predict(self, X, normalize=True, use_biases=True):      
        if normalize:
            X = self.__normalize(X)

        if use_biases:
            X = self.__use_biases(X)

        return self.__softmax(np.dot(X, self.params))


    def get_params(self):
        return self.params


    def _cross_entropy_loss(self):
        y_hat = self.__softmax(np.dot(self.X, self.params))
        return -np.mean(np.sum(self.y * np.log(y_hat) + (1 - self.y) * np.log(1 - y_hat), axis=1))


    def __gradients(self):
        return self.X.T.dot(self.__softmax(self.X.dot(self.params)) - self.y)


    def __softmax(self, x):
        e = np.exp(x - np.max(x))
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:  
            return e / np.array([np.sum(e, axis=1)]).T


    def __normalize(self, X):
        return (X - X.mean()) / X.std()


    def __use_biases(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
