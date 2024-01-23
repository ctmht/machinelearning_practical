from sklearn.tree import DecisionTreeClassifier


class Baseline:
    def __init__(self, max_leaf_nodes=20):
        self.model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
