import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.embedder import Embedder
from src.model_interface import ModelInterface
from src.processor import Processor


class DecisionTree(ModelInterface):
    """ Decision Tree Classifier: Baseline model for emoji prediction task """

    def __init__(self, processor: Processor, embedder: Embedder, **kwargs):
        super().__init__(processor, embedder)

        self.model = DecisionTreeClassifier(**kwargs)

        self.max_emb = 140 # Max num of 'words' in one tweet, I guess

    def train(self, train_text: pd.DataFrame, train_labels: pd.DataFrame):
        super()._train(
            self._get_uniform_embeddings(train_text),
            train_labels
        )

    def predict(self, predict_text: pd.DataFrame):
        return super()._predict(
            self._get_uniform_embeddings(predict_text)
        )

    @staticmethod
    def evaluate(pred_out: np.array, true_out: np.array, metrics):
        evals = []

        for metric in metrics:
            try:
                evals.append(metric(pred_out, true_out, average = "macro"))
            except TypeError:
                evals.append(metric(pred_out, true_out))

        return evals

    def _get_uniform_embeddings(self, wordlists: pd.DataFrame):
        return [
            np.array(
                self.embedder.uniform_tweet_embedding(
                    wordlist, size = self.max_emb
                )
            ) for wordlist in wordlists
        ]