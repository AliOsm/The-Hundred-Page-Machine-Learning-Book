import pprint
import pandas as pd


from decision_tree import DecisionTree


def load_car():
    dataset = pd.read_csv('car.csv', header=None)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    X = dataset.iloc[:,0:6].to_numpy()
    y = dataset.iloc[:,6].to_numpy()

    return X, y


if __name__ == '__main__':
    X, y = load_car()

    decision_tree = DecisionTree(X, y)
    decision_tree.build()
    print(X[0], y[0])
    print(decision_tree.predict(X[0]))

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(decision_tree.tree)
