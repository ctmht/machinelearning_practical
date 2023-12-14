import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot


def create_tweet_embeddings(word2vec, tweets):
    empty_embedding = np.zeros(word2vec.vector_size)

    tweet_embeddings = []
    for tweet in tweets:
        if tweet:
            embeddings = []
            for word in tweet:
                if word in word2vec.wv:
                    embeddings.append(word2vec.wv[word])
                else:
                    embeddings.append(empty_embedding)
            embeddings = np.array(embeddings)
            tweet_embeddings.append(embeddings.mean(axis=0))
        else:
            tweet_embeddings.append(empty_embedding)

    return np.array(tweet_embeddings)


def create_tweet_embeddings_sequential(word2vec, tweets):
    empty_embedding = np.zeros(word2vec.vector_size)

    tweet_embeddings = []
    for tweet in tweets:
        if tweet:
            embeddings = [word2vec.wv[word] if word in word2vec.wv else empty_embedding for word in tweet]
            tweet_embeddings.append(embeddings)
        else:
            tweet_embeddings.append([empty_embedding])

    tweet_embeddings = pad_sequences(tweet_embeddings, maxlen=50, dtype='float32', padding='post', truncating='post')
    return np.array(tweet_embeddings)


def create_tweet_encodings(word2vec, tweets):
    vocab_size = word2vec.wv.vectors.shape[0]
    for idx in range(len(tweets)):
        tweets[idx] = ' '.join(tweets[idx])
    encoded = [one_hot(tweet, vocab_size) for tweet in tweets]
    return pad_sequences(encoded, maxlen=50, dtype='float32', padding='post', truncating='post')
