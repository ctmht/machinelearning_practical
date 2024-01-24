# from preprocess import preprocess
# from embed import embed
# from preprocessing.embeddings import create_word_to_int_map
# from preprocessing.text_preprocessing import longest_preprocessed_tweet
# from gensim.models import Word2Vec
# from hyper_model import initialize_and_train_lstm
# from saving_loading import save, load
#
#
# def create_word2vec_and_word2int(train_tweets, save_data=False, load_data=False):
#     if not load_data:
#         w2v_model = Word2Vec(train_tweets, min_count=1)
#         w2i_map = create_word_to_int_map(train_tweets)
#     else:
#         w2v_model = load("../models/w2v_model.pkl")
#         w2i_map = load("../models/w2i_map.pkl")
#
#     if save_data:
#         save(w2v_model, '../models/w2v_model.pkl')
#         save(w2i_map, '../models/w2i_map.pkl')
#
#     return w2v_model, w2i_map, longest_preprocessed_tweet(train_tweets)
#
#
# def hyper_parameter_search():
#     train_tweets, train_labels, train_labels_one_hot, val_tweets, val_labels, val_labels_one_hot \
#         = preprocess(save_data=False, load_data=True)
#     w2v_model, w2i_map, max_tweet_len = create_word2vec_and_word2int(train_tweets, save_data=False, load_data=True)
#     train_tweet_mean_embeddings, \
#         val_tweet_mean_embeddings, \
#         train_tweet_padded_embeddings, \
#         val_tweet_padded_embeddings, \
#         embedding_matrix \
#         = embed(w2v_model, w2i_map, max_tweet_len, train_tweets, val_tweets, save_data=False, load_data=True)
#
#     model = initialize_and_train_lstm(embedding_matrix, max_tweet_len,
#                                       train_tweet_padded_embeddings, train_labels_one_hot,
#                                       save_data=True, load_data=False)
#
#     loss, accuracy = model.evaluate(val_tweet_padded_embeddings, val_labels_one_hot)
#     print('Accuracy model: \t%f' % (accuracy * 100))
#
#
# if __name__ == '__main__':
#     hyper_parameter_search()
