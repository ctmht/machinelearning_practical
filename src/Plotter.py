from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from typing import List, Dict, Any
import numpy as np


def _plot_matrix(test_labels: List[int], predicted_labels: List[int]):
    # Plots the confusion matrix of the evaluation
    matrix = confusion_matrix(test_labels, predicted_labels)
    heatmap(matrix)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def _plot_accuracy(history):
    # History is an object returned by model.fit
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xticks([4, 9, 14, 19], ['5', '10', '15', '20'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


class Plotter:
    def __init__(self, model):
        self._model = model
