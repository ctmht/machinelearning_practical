from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re


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
    tweet = remove_numbers(tweet)
    tweet = remove_punctuation(tweet)
    tweet = remove_whitespace(tweet)
    tweet = remove_stopwords(tweet)
    return tweet


def load_text(file_name):
    tweets = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for tweet in file:
            tweet = preprocess_tweet(tweet)
            tweets.append(tweet)
    return tweets
