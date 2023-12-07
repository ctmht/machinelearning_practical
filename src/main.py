from preprocessing.text_preprocessing import preprocess_tweets
from preprocessing.embeddings import create_tweet_embeddings, create_word_embeddings


tweets = preprocess_tweets("../../data/train_text.txt")
tweet_vectors = create_tweet_embeddings(tweets)
