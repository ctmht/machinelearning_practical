from collections.abc import Callable

import numpy as np

from src.embedder import Embedder
from src.processor import Processor


class ModelInterface:
    """ Interface for supervised models in the emoji prediction task """

    def __init__(self, processor: Processor, embedder: Embedder):
        """
        Initialize a generic model with access to a processor and an embedder
        """
        self.processor: Processor = processor
        self.embedder: Embedder = embedder

        self.model = None

    def _train(self, indata, labels, **kwargs):
        """
        Train the supervised model on input data with labeled outputs
        Args:
            indata: input data
            labels: labeled output data
            **kwargs: extra params for model-specific fitting function
        """
        self.model.fit(X = indata, y = labels, **kwargs)

    def _predict(self, indata):
        """
        Predict classes based on given inputs
        Args:
            indata: input data
        Return:
            : list of predicted labels
        """
        return self.model.predict(indata)

    @staticmethod # maybe redundant? idk
    def _evaluate_prediction(
            pred_out: np.array,
            true_out: np.array,
            scoring
        ):
        """
        Use scoring to assess classification quality
        Args:
            pred_out: predicted outputs
            true_out: true outputs
            scoring: function to compute classification evaluation
        """
        return scoring(pred_out, true_out)