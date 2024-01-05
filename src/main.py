import pandas as pd

from src.processor import Processor
from src.embedder import Embedder
from src.decision_tree import DecisionTree


def main():
    # LOAD DATA
    load_from_files = False     # load from files if True, preprocess if False
    data_folder = "../data/"

    # Preprocessing
    processor = Processor(data_folder)
    data = processor.get_data(from_files = load_from_files)
    df = pd.DataFrame(data)
    print("\nData:\n", df, end="\n\n")

    # Embeddings
    embedder = Embedder(data_folder)
    embedder.create_embeddings(df["text"])
    embedder.save_model()

    # Baseline
    dt_hpar = {
        "criterion": "entropy",
        "max_depth": 20,
        "max_leaf_nodes": 1000
    }
    baseline = DecisionTree(processor, embedder, **dt_hpar)
    baseline.train(df)

    # TODO: Baseline and LSTM classes with docs, inheriting from Model

    # Save to memory and create embeddings
    # print("Creating embeddings")
    # embeddings = Embeddings("../data/prc_data/")
    # embeddings.embed_words("train")

    # # Create embeddings
    # print("embeddings")
    # w2v_model = Word2Vec(train_tweets, min_count=1)
    #
    # # Initialize and train baseline and LSTM model
    # print("baseline")
    # baseline = Baseline()
    # baseline.train(create_tweet_embeddings(w2v_model, train_tweets),\
    # train_labels)
    #
    # print("lstm")
    # model = LSTM(w2v_model, 50, 20)
    # print(model.model.summary())
    # model.train(create_tweet_encodings(w2v_model, train_tweets),\
    # train_labels)
    #
    # # Evaluate both models
    # print("evaluate")
    # baseline_pred = baseline.predict(create_tweet_embeddings(w2v_model,\
    # val_tweets))
    # loss, accuracy = model.evaluate(create_tweet_encodings(w2v_model,\
    # val_tweets), val_labels)
    # print('Accuracy baseline: \t%f' % (np.sum(baseline_pred == val_labels)\
    # / val_labels.size * 100))
    # print('Accuracy model: \t%f' % (accuracy * 100))
    # print('Random guess: \t%f' % (np.max(np.unique(val_labels,\
    # return_counts=True)[1]) / val_labels.size * 100))


if __name__ == '__main__':
    main()
