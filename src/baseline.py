from sklearn.tree import DecisionTreeClassifier
from saving_loading import save, load


class Baseline:
    def __init__(self, max_leaf_nodes=20):
        self.model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def initialize_and_train_baseline(train_tweet_mean_embeddings, train_labels, save_data=False, load_data=False):
    print("Baseline")
    if not load_data:
        baseline = Baseline()
        baseline.train(train_tweet_mean_embeddings, train_labels)
    else:
        baseline = load('../models/baseline_model.pkl')

    if save_data:
        save(baseline, '../models/baseline_model.pkl')
    return baseline
