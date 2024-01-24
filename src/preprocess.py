from preprocessing.text_preprocessing import *
from saving_loading import save, load


def preprocessed_sample(tweets, sample_size=50):
    with open("../data/preprocessed_tweets.txt", 'w') as file:
        for tweet in tweets[:sample_size]:
            file.write(' '.join(tweet) + '\n')


def preprocess(save_data=False, load_data=False):
    print("Preprocess")

    if not load_data:
        train_tweets = preprocess_tweets("../data/train_text.txt")
        train_labels = preprocess_labels("../data/train_labels.txt")
        train_labels_one_hot = preprocess_labels_one_hot("../data/train_labels.txt")

        val_tweets = preprocess_tweets("../data/val_text.txt")
        val_labels = preprocess_labels("../data/val_labels.txt")
        val_labels_one_hot = preprocess_labels_one_hot("../data/val_labels.txt")

        preprocessed_sample(train_tweets)
    else:
        train_tweets, train_labels, train_labels_one_hot = load('../processed_data/train.pkl')
        val_tweets, val_labels, val_labels_one_hot = load('../processed_data/val.pkl')

    if save_data:
        save((train_tweets, train_labels, train_labels_one_hot), '../processed_data/train.pkl')
        save((val_tweets, val_labels, val_labels_one_hot), '../processed_data/val.pkl')

    return train_tweets, train_labels, train_labels_one_hot, val_tweets, val_labels, val_labels_one_hot
