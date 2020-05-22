import pandas as pd
import matplotlib.pyplot as plt


from logistic_regression import LogisticRegression


def load_iris():
    dataset = pd.read_csv('iris.csv')
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    X = dataset.iloc[:,0:4].to_numpy()
    y = dataset.iloc[:,4]
    for i, c in enumerate(sorted(set(y))):
        y[y == c] = i
    y = pd.get_dummies(y).to_numpy()

    return X, y


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y = load_iris()

    logistic_regression = LogisticRegression(X, y)
    losses = logistic_regression.fit()

    plot_losses(losses)
