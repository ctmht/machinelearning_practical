import os
import numpy as np
from preprocessing.text_preprocessing import preprocess_tweets, preprocess_labels
from preprocessing.embeddings import *
from gensim.models import Word2Vec
from baseline import Baseline
from model import LSTM


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
train_text_path = os.path.join(data_dir, "train_text.txt")
train_labels_path = os.path.join(data_dir, "train_labels.txt")
val_text_path = os.path.join(data_dir, "val_text.txt")
val_labels_path = os.path.join(data_dir, "val_labels.txt")


def main():

    # Preprocess the data
    print("preprocess")
    train_tweets = preprocess_tweets("../data/train_text.txt")
    train_labels = preprocess_labels("../data/train_labels.txt")
    val_tweets = preprocess_tweets("../data/val_text.txt")
    val_labels = preprocess_labels("../data/val_labels.txt")

    # Create embeddings
    print("embeddings")
    w2v_model = Word2Vec(train_tweets, min_count=1)

    # Initialize and train baseline and LSTM model
    print("baseline")
    baseline = Baseline()
    baseline.train(create_tweet_embeddings(w2v_model, train_tweets), train_labels)

    print("lstm")
    model = LSTM(w2v_model, 50, 20)
    print(model.model.summary())
    model.train(create_tweet_encodings(w2v_model, train_tweets), train_labels)

    # Evaluate both models
    print("evaluate")
    baseline_pred = baseline.predict(create_tweet_embeddings(w2v_model, val_tweets))
    print('Accuracy baseline: \t%f' % (np.sum(baseline_pred == val_labels) / val_labels.size * 100))
    loss, accuracy = model.evaluate(create_tweet_encodings(w2v_model, val_tweets), val_labels)
    print('Accuracy model: \t%f' % (accuracy * 100))


if __name__ == '__main__':
    main()
