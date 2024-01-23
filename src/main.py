import os
from sklearn.utils import class_weight
from preprocessing.text_preprocessing import *
from preprocessing.embeddings import *
from gensim.models import Word2Vec
from baseline import Baseline
from model import LSTM
from saving_loading import save


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
train_text_path = os.path.join(data_dir, "train_text.txt")
train_labels_path = os.path.join(data_dir, "train_labels.txt")
val_text_path = os.path.join(data_dir, "val_text.txt")
val_labels_path = os.path.join(data_dir, "val_labels.txt")


def main():

    # Preprocessing
    print("preprocess")
    train_tweets = preprocess_tweets("../data/train_text.txt")
    with open("../data/preprocessed_tweets.txt", 'w') as file:
        for tweet in train_tweets[:50]:
            file.write(' '.join(tweet) + '\n')
    train_labels = preprocess_labels("../data/train_labels.txt")
    train_labels_one_hot = preprocess_labels_one_hot("../data/train_labels.txt")

    val_tweets = preprocess_tweets("../data/val_text.txt")
    val_labels = preprocess_labels("../data/val_labels.txt")
    val_labels_one_hot = preprocess_labels_one_hot("../data/val_labels.txt")

    # Create word2vec and word2int
    w2v_model = Word2Vec(train_tweets, min_count=1)
    w2i_map = create_word_to_int_map(train_tweets)
    max_tweet_len = longest_preprocessed_tweet(train_tweets)

    # Create embeddings for different models
    print("embeddings")
    train_tweet_mean_embeddings = create_mean_tweet_embeddings(w2v_model, train_tweets)
    val_tweet_mean_embeddings = create_mean_tweet_embeddings(w2v_model, val_tweets)
    train_tweet_padded_embeddings = create_padded_tweet_embeddings(train_tweets, w2i_map, max_tweet_len)
    val_tweet_padded_embeddings = create_padded_tweet_embeddings(val_tweets, w2i_map, max_tweet_len)
    embedding_matrix = create_embedding_matrix(w2v_model, w2i_map)

    # Initialize and train baseline
    print("baseline")
    baseline = Baseline()
    baseline.train(train_tweet_mean_embeddings, train_labels)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    # Initialize and train LSTM model
    print("lstm")
    model = LSTM(embedding_matrix, max_tweet_len, num_classes=20)
    print(model.model.summary())
    model.train(train_tweet_padded_embeddings, train_labels_one_hot, 20, 32)

    # Evaluation
    print('Random guess: \t%f' % (np.max(np.unique(val_labels, return_counts=True)[1]) / val_labels.size * 100))
    baseline_pred = baseline.predict(val_tweet_mean_embeddings)
    print('Accuracy baseline: \t%f' % (np.sum(baseline_pred == val_labels) / val_labels.size * 100))
    loss, accuracy = model.evaluate(val_tweet_padded_embeddings, val_labels_one_hot)
    print('Accuracy model: \t%f' % (accuracy * 100))

    save(baseline, '../models/baseline_model.pkl')
    save(model, '../models/lstm_model.pkl')
    save(w2v_model, '../models/w2v_model.pkl')
    save(w2i_map, '../models/w2i_map.pkl')


if __name__ == '__main__':
    main()
