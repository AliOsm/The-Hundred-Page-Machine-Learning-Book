import numpy as np


eps = np.finfo(float).eps


class DecisionTree():
    def __init__(self, X, y, max_depth=4, epsilon=1e-6):
        assert(X.shape[0] == len(y))

        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.tree = {}


    def build(self):
        self.tree = self.__build(self.X, self.y, {}, 0)


    def predict(self, x):
        return self.__predict(x, self.tree)


    def get_tree(self):
        return self.tree


    def __build(self, X, y, tree, depth):
        if depth >= self.max_depth:
            return None

        attribute = self.__find_best_attribute(X, y)

        if attribute is None:
            return None

        values = sorted(set(X[:, attribute]))

        for v in values:
            split_X, split_y = self.__split_table(X, y, attribute, v)

            if len(set(split_y)) == 1:
                if attribute not in tree:
                    tree[attribute] = {}
                tree[attribute][v] = split_y[0]
            else:
                temp = self.__build(split_X, split_y, {}, depth + 1)
                if temp is None:
                    continue
                if attribute not in tree:
                    tree[attribute] = {}
                tree[attribute][v] = temp

        if tree == {}:
            tree = None
        return tree


    def __predict(self, x, tree):
        for attribute in tree.keys():
            value = x[attribute]
            if attribute not in tree or value not in tree[attribute]:
                return None
            tree = tree[attribute][value]
            prediction = 0

            if type(tree) is dict:
                prediction = self.__predict(x, tree)
            else:
                prediction = tree
                break

        return prediction


    def __find_best_attribute(self, X, y):
        entropy = self.__calc_entropy(y)
        mx_i = -1
        mx_v = -1

        for i in range(X.shape[1]):
            temp = entropy - self.__calc_attribute_entropy(X[:, i], y)
            if temp > mx_v:
                mx_v = temp
                mx_i = i

        if mx_v < self.epsilon:
            return None

        return mx_i


    def __calc_entropy(self, y):
        entropy = 0
        classes = sorted(set(y))
        
        for c in classes:
            fraction = len(y[y == c]) / len(y)
            entropy += -fraction * np.log2(fraction)
        
        return entropy


    def __calc_attribute_entropy(self, x, y):
        entropy = 0
        values = sorted(set(x))
        classes = sorted(set(y))

        for v in values:
            temp = 0
            for c in classes:
                a = sum((x == v) & (y == c))
                b = len(x[x == v])
                fraction = a / (b + eps)
                temp += -fraction * np.log2(fraction + eps)
            entropy += -(b / len(x)) * temp

        return abs(entropy)


    def __split_table(self, X, y, attribute, value):
        needed = X[:, attribute] == value
        return X[needed], y[needed]
