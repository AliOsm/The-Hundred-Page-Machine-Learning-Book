import pandas as pd
import matplotlib.pyplot as plt


from linear_regression import LinearRegression


def load_boston():
    dataset = pd.read_csv('boston.csv')

    X = dataset.iloc[:,0:13].to_numpy()
    y = dataset.iloc[:,13].to_numpy()

    return X, y


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y = load_boston()

    linear_regression = LinearRegression(X, y)
    losses = linear_regression.fit()

    plot_losses(losses)
