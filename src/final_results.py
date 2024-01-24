import os
from preprocessing.text_preprocessing import *
from preprocessing.embeddings import *
from preprocessing.embeddings import create_word_to_int_map
from preprocessing.text_preprocessing import longest_preprocessed_tweet
from gensim.models import Word2Vec
from baseline import initialize_and_train_baseline_final
from model import initialize_and_train_lstm
from hyper_model import initialize_and_train_hyper_lstm
from saving_loading import save, load
from sklearn.metrics import f1_score


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
train_text_path = os.path.join(data_dir, "train_text.txt")
train_labels_path = os.path.join(data_dir, "train_labels.txt")
val_text_path = os.path.join(data_dir, "val_text.txt")
val_labels_path = os.path.join(data_dir, "val_labels.txt")


def preprocess(save_data=False, load_data=False):
    print("Preprocess")

    if not load_data:
        tv_tweets = preprocess_tweets("../data/train_val_text.txt")
        tv_labels = preprocess_labels("../data/train_val_labels.txt")
        tv_labels_one_hot = preprocess_labels_one_hot("../data/train_val_labels.txt")

        test_tweets = preprocess_tweets("../data/test_text.txt")
        test_labels = preprocess_labels("../data/test_labels.txt")
        test_labels_one_hot = preprocess_labels_one_hot("../data/test_labels.txt")

    else:
        tv_tweets, tv_labels, tv_labels_one_hot = load('../processed_data/tv.pkl')
        test_tweets, test_labels, test_labels_one_hot = load('../processed_data/test.pkl')

    if save_data:
        save((tv_tweets, tv_labels, tv_labels_one_hot), '../processed_data/tv.pkl')
        save((test_tweets, test_labels, test_labels_one_hot), '../processed_data/test.pkl')

    return tv_tweets, tv_labels, tv_labels_one_hot, test_tweets, test_labels, test_labels_one_hot


def embed(w2v_model, w2i_map, max_tweet_len, train_tweets, val_tweets, save_data=False, load_data=False):
    print("Embeddings")

    if not load_data:
        train_tweet_mean_embeddings = create_mean_tweet_embeddings(w2v_model, train_tweets)
        val_tweet_mean_embeddings = create_mean_tweet_embeddings(w2v_model, val_tweets)

        train_tweet_padded_embeddings = create_padded_tweet_embeddings(train_tweets, w2i_map, max_tweet_len)
        val_tweet_padded_embeddings = create_padded_tweet_embeddings(val_tweets, w2i_map, max_tweet_len)

        embedding_matrix = create_embedding_matrix(w2v_model, w2i_map)
    else:
        train_tweet_mean_embeddings, val_tweet_mean_embeddings = \
            load('../processed_data/mean_embeddings_final.pkl')
        train_tweet_padded_embeddings, val_tweet_padded_embeddings = \
            load('../processed_data/padded_embeddings_final.pkl')
        embedding_matrix = load('../processed_data/embedding_matrix_final.pkl')

    if save_data:
        save((train_tweet_mean_embeddings, val_tweet_mean_embeddings),
             '../processed_data/mean_embeddings_final.pkl')
        save((train_tweet_padded_embeddings, val_tweet_padded_embeddings),
             '../processed_data/padded_embeddings_final.pkl')
        save(embedding_matrix, '../processed_data/embedding_matrix_final.pkl')

    return train_tweet_mean_embeddings, \
        val_tweet_mean_embeddings, \
        train_tweet_padded_embeddings, \
        val_tweet_padded_embeddings, \
        embedding_matrix


def create_word2vec_and_word2int(train_tweets, save_data=False, load_data=False):
    if not load_data:
        w2v_model = Word2Vec(train_tweets, min_count=1)
        w2i_map = create_word_to_int_map(train_tweets)
    else:
        w2v_model = load("../models/w2v_model_final.pkl")
        w2i_map = load("../models/w2i_map_final.pkl")

    if save_data:
        save(w2v_model, '../models/w2v_model_final.pkl')
        save(w2i_map, '../models/w2i_map_final.pkl')

    return w2v_model, w2i_map, longest_preprocessed_tweet(train_tweets)


def main():
    # preprocess
    tv_tweets, tv_labels, tv_labels_one_hot, test_tweets, test_labels, test_labels_one_hot \
        = preprocess(save_data=True, load_data=False)

    w2v_model, w2i_map, max_tweet_len = create_word2vec_and_word2int(tv_tweets, save_data=True, load_data=False)

    # create embeddings
    tv_tweet_mean_embeddings, \
        test_tweet_mean_embeddings, \
        tv_tweet_padded_embeddings, \
        test_tweet_padded_embeddings, \
        embedding_matrix \
        = embed(w2v_model, w2i_map, max_tweet_len, tv_tweets, test_tweets, save_data=True, load_data=False)

    # create and train models
    baseline = initialize_and_train_baseline_final(tv_tweet_mean_embeddings, tv_labels,
                                                   save_data=True, load_data=False)

    model = initialize_and_train_lstm(embedding_matrix, max_tweet_len, 100, 20, 32, 'lstm',
                                      tv_tweet_padded_embeddings, tv_labels_one_hot,
                                      save_data=True, load_data=False)

    hyper_model = initialize_and_train_lstm(embedding_matrix, max_tweet_len, 240, 50, 64, 'hyper',
                                            tv_tweet_padded_embeddings, tv_labels_one_hot,
                                            save_data=True, load_data=False)

    # evaluate models
    print(f'Random guess:         {np.max(np.unique(test_labels, return_counts=True)[1]) / test_labels.size * 100}')
    baseline_pred = baseline.predict(test_tweet_mean_embeddings)
    print(f'Accuracy baseline:    {np.sum(baseline_pred == test_labels) / test_labels.size * 100}')
    loss, accuracy = model.evaluate(test_tweet_padded_embeddings, test_labels_one_hot)
    print(f'Accuracy model:       {accuracy * 100}')
    loss, accuracy = hyper_model.evaluate(test_tweet_padded_embeddings, test_labels_one_hot)
    print(f'Accuracy hyper model: {accuracy * 100}')

    raw_predictions = model.predict(test_tweet_padded_embeddings)
    label_predictions = np.array([], dtype=int)
    for prediction in raw_predictions:
        label_predictions = np.append(label_predictions, np.argmax(prediction))

    hp_raw_predictions = hyper_model.predict(test_tweet_padded_embeddings)
    hp_label_predictions = np.array([], dtype=int)
    for prediction in hp_raw_predictions:
        hp_label_predictions = np.append(hp_label_predictions, np.argmax(prediction))

    print("true:       ", test_labels[:20])
    print("baseline:   ", baseline_pred[:20])
    print("model:      ", label_predictions[:20])
    print("hyper model:", hp_label_predictions[:20])

    # Calculate F1 scores
    most_frequent_class = np.argmax(np.unique(test_labels, return_counts=True)[1])
    f1_macro = f1_score(test_labels, [most_frequent_class for _ in test_labels], average='macro')
    print(f'Random Macro F1 Score:   {f1_macro * 100}')

    f1_macro = f1_score(test_labels, baseline_pred, average='macro')
    print(f'Baseline Macro F1 Score: {f1_macro * 100}')

    f1_macro = f1_score(test_labels, label_predictions, average='macro')
    print(f'Model Macro F1 Score:    {f1_macro * 100}')

    f1_macro = f1_score(test_labels, hp_label_predictions, average='macro')
    print(f'Hyper Macro F1 Score:    {f1_macro * 100}')


if __name__ == '__main__':
    main()
