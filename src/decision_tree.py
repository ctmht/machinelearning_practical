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

        self.max_emb = 140

    def train(self, data: pd.DataFrame):
        print("\nTraining Decision Tree Model\n")

        train_df = data.query("type == 'train'")

        uniform_embeddings = [
            np.array(
                self.embedder.uniform_tweet_embedding(
                    wordlist, size = self.max_emb
                )
            ) for wordlist in train_df["text"]
        ]

        super()._train(
            uniform_embeddings,
            train_df["label"]
        )

    def predict(self):
        pass

    def evaluate(self):
        pass