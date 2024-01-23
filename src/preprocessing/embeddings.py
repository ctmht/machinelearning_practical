import numpy as np
from keras.preprocessing.sequence import pad_sequences


def create_mean_tweet_embeddings(word2vec, tweets):
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


def create_word_to_int_map(tweets):
    idx = 1
    w2i_map = {'_': 0}
    for tweet in tweets:
        for word in tweet:
            if word not in w2i_map:
                w2i_map[word] = idx
                idx += 1
    return w2i_map


def create_padded_tweet_embeddings(tweets, w2i_map, max_tweet_len):
    encoded = []
    for tweet in tweets:
        encoded_tweet = []
        for word in tweet:
            if word in w2i_map:
                encoded_tweet.append(w2i_map[word])
        encoded.append(encoded_tweet)
    return pad_sequences(encoded, maxlen=max_tweet_len, dtype='float32', padding='post', truncating='post')


def create_embedding_matrix(w2v_model, w2i_map):
    vocab_size = w2v_model.wv.vectors.shape[0] + 1  # 1 empty vector used for padding
    embedding_dim = w2v_model.wv.vectors.shape[1]
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in w2i_map.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    return embedding_matrix
