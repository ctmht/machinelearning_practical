from preprocessing.embeddings import *
from saving_loading import save, load


def embed(w2v_model, w2i_map, max_tweet_len, train_tweets, val_tweets, save_data=False, load_data=False):
    print("Embeddings")

    if not load_data:
        train_tweet_mean_embeddings = create_mean_tweet_embeddings(w2v_model, train_tweets)
        val_tweet_mean_embeddings = create_mean_tweet_embeddings(w2v_model, val_tweets)

        train_tweet_padded_embeddings = create_padded_tweet_embeddings(train_tweets, w2i_map, max_tweet_len)
        val_tweet_padded_embeddings = create_padded_tweet_embeddings(val_tweets, w2i_map, max_tweet_len)

        embedding_matrix = create_embedding_matrix(w2v_model, w2i_map)
    else:
        train_tweet_mean_embeddings, val_tweet_mean_embeddings = load('../processed_data/mean_embeddings.pkl')
        train_tweet_padded_embeddings, val_tweet_padded_embeddings = load('../processed_data/padded_embeddings.pkl')
        embedding_matrix = load('../processed_data/embedding_matrix.pkl')

    if save_data:
        save((train_tweet_mean_embeddings, val_tweet_mean_embeddings), '../processed_data/mean_embeddings.pkl')
        save((train_tweet_padded_embeddings, val_tweet_padded_embeddings), '../processed_data/padded_embeddings.pkl')
        save(embedding_matrix, '../processed_data/embedding_matrix.pkl')

    return train_tweet_mean_embeddings, \
        val_tweet_mean_embeddings, \
        train_tweet_padded_embeddings, \
        val_tweet_padded_embeddings, \
        embedding_matrix
