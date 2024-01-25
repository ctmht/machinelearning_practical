import os
import numpy as np
from preprocess import preprocess
from embed import embed
from preprocessing.embeddings import create_word_to_int_map
from preprocessing.text_preprocessing import longest_preprocessed_tweet
from gensim.models import Word2Vec
from baseline import initialize_and_train_baseline
from model import initialize_and_train_lstm
from Plotter import _plot_matrix, _plot_accuracy
from saving_loading import save, load
from sklearn.metrics import f1_score


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
train_text_path = os.path.join(data_dir, "train_text.txt")
train_labels_path = os.path.join(data_dir, "train_labels.txt")
val_text_path = os.path.join(data_dir, "val_text.txt")
val_labels_path = os.path.join(data_dir, "val_labels.txt")


def create_word2vec_and_word2int(train_tweets, save_data=False, load_data=False):
    if not load_data:
        w2v_model = Word2Vec(train_tweets, min_count=1)
        w2i_map = create_word_to_int_map(train_tweets)
    else:
        w2v_model = load("../models/w2v_model.pkl")
        w2i_map = load("../models/w2i_map.pkl")

    if save_data:
        save(w2v_model, '../models/w2v_model.pkl')
        save(w2i_map, '../models/w2i_map.pkl')

    return w2v_model, w2i_map, longest_preprocessed_tweet(train_tweets)


def main():
    # preprocess
    train_tweets, train_labels, train_labels_one_hot, val_tweets, val_labels, val_labels_one_hot \
        = preprocess(save_data=False, load_data=True)

    w2v_model, w2i_map, max_tweet_len = create_word2vec_and_word2int(train_tweets, save_data=False, load_data=True)

    # create embeddings
    train_tweet_mean_embeddings, \
        val_tweet_mean_embeddings, \
        train_tweet_padded_embeddings, \
        val_tweet_padded_embeddings, \
        embedding_matrix \
        = embed(w2v_model, w2i_map, max_tweet_len, train_tweets, val_tweets, save_data=False, load_data=True)

    # create and train models
    baseline = initialize_and_train_baseline(train_tweet_mean_embeddings, train_labels,
                                             save_data=False, load_data=True)

    model = initialize_and_train_lstm(embedding_matrix, max_tweet_len, 100, 20, 32, 'lstm',
                                      train_tweet_padded_embeddings, train_labels_one_hot,
                                      (val_tweet_padded_embeddings, val_labels_one_hot),
                                      save_data=False, load_data=True)

    hyper_model = initialize_and_train_lstm(embedding_matrix, max_tweet_len, 240, 50, 64, 'hyper',
                                            train_tweet_padded_embeddings, train_labels_one_hot,
                                            save_data=False, load_data=True)

    # evaluate models
    print(f'Random guess:         {np.max(np.unique(val_labels, return_counts=True)[1]) / val_labels.size * 100}')
    baseline_pred = baseline.predict(val_tweet_mean_embeddings)
    print(f'Accuracy baseline:    {np.sum(baseline_pred == val_labels) / val_labels.size * 100}')
    loss, accuracy = model.evaluate(val_tweet_padded_embeddings, val_labels_one_hot)
    print(f'Accuracy model:       {accuracy * 100}')
    loss, accuracy = hyper_model.evaluate(val_tweet_padded_embeddings, val_labels_one_hot)
    print(f'Accuracy hyper model: {accuracy * 100}')

    raw_predictions = model.predict(val_tweet_padded_embeddings)
    label_predictions = np.array([], dtype=int)
    for prediction in raw_predictions:
        label_predictions = np.append(label_predictions, np.argmax(prediction))

    hp_raw_predictions = hyper_model.predict(val_tweet_padded_embeddings)
    hp_label_predictions = np.array([], dtype=int)
    for prediction in hp_raw_predictions:
        hp_label_predictions = np.append(hp_label_predictions, np.argmax(prediction))

    print("true:       ", val_labels[:20])
    print("baseline:   ", baseline_pred[:20])
    print("model:      ", label_predictions[:20])
    print("hyper model:", hp_label_predictions[:20])

    # Calculate F1 scores
    most_frequent_class = np.argmax(np.unique(val_labels, return_counts=True)[1])
    f1_macro = f1_score(val_labels, [most_frequent_class for _ in val_labels], average='macro')
    print(f'Random Macro F1 Score:   {f1_macro * 100}')

    f1_macro = f1_score(val_labels, baseline_pred, average='macro')
    print(f'Baseline Macro F1 Score: {f1_macro * 100}')

    f1_macro = f1_score(val_labels, label_predictions, average='macro')
    print(f'Model Macro F1 Score:    {f1_macro * 100}')

    f1_macro = f1_score(val_labels, hp_label_predictions, average='macro')
    print(f'Hyper Macro F1 Score:    {f1_macro * 100}')

    _plot_accuracy(load('../models/train_history_with_validation.pkl'))


if __name__ == '__main__':
    main()
