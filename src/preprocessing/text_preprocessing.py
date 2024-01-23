from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordsegment import segment
from wordsegment import load as word_segment_load
import string
import re
import numpy as np

word_segment_load()  # for wordsegment to work


def split_word(word: str) -> str:
    split_words = segment(word)
    return " ".join(split_words)


def split_hashtags(text: str) -> str:
    pattern = r'#(\w+)'
    words = text.split()
    for word in words:
        if re.match(pattern, word):
            split_words = split_word(word)
            text = text.replace(word, split_words)
    return text


def remove_at(text):
    text = text.replace('@user', '')
    result = re.sub(r'@ ?[A-Za-z]+', '', text)
    return result


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_whitespace(text):
    return " ".join(text.split())


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    cleaned_text = [word for word in filtered_text if word != '️']
    for idx, word in enumerate(cleaned_text):
        word = word.replace('…', '')
        cleaned_text[idx] = word
    return cleaned_text


def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = split_hashtags(tweet)
    tweet = remove_at(tweet)
    tweet = remove_numbers(tweet)
    tweet = remove_punctuation(tweet)
    tweet = remove_whitespace(tweet)
    tweet = remove_stopwords(tweet)
    return tweet


def preprocess_tweets(file_name):
    tweets = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for tweet in file:
            tweet = preprocess_tweet(tweet)
            tweets.append(tweet)
    return tweets


# Preprocess the labels
def preprocess_labels(file_name):
    labels = []
    with open(file_name, "r", encoding="utf-8") as file:
        for label in file:
            label = int(label.strip())
            labels.append(label)
    return np.array(labels)


def preprocess_labels_one_hot(file_name):
    labels = []
    with open(file_name, "r", encoding="utf-8") as file:
        for label in file:
            label = int(label.strip())
            one_hot = np.zeros(20)
            one_hot[label] = 1
            labels.append(one_hot)
    return np.array(labels)


def longest_preprocessed_tweet(tweets):
    max_tweet_len = 5
    for tweet in tweets:
        if len(tweet) > max_tweet_len:
            max_tweet_len = len(tweet)
    return max_tweet_len
