from preprocessing import *

tweets = text_preprocessing.preprocess_tweets("../../data/train_text.txt")
tweet_vectors = embeddings.create_tweet_embeddings(tweets)
