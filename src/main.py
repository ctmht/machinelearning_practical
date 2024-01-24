import pandas as pd
import sklearn.metrics as skmt

from src.processor import Processor
from src.embedder import Embedder
from src.decision_tree import DecisionTree
from src.util import load


def main():
    # LOAD DATA
    load_from_files = True     # load from files if True, preprocess if False
    data_folder = "../data/"

    print("Preprocessing")
    # Preprocessing
    processor = Processor(data_folder)
    data = processor.get_data(
        from_files = load_from_files,
        augmented = True
    )
    df = pd.DataFrame(data)
    train_df = df.query("type == 'train'")
    val_df = df.query("type == 'val'")
    test_df = df.query("type == 'test'")
    print(df)

    print("Creating embeddings")
    # Embeddings
    embedder = Embedder(data_folder)
    embedder.train_embeddings(df["text"])
    embedder.save_model()

    # Metrics
    metrics = [
        skmt.accuracy_score,
        skmt.f1_score,
        skmt.confusion_matrix
    ]
    metric_names = [
        "Accuracy", "F1-score", "Confusion Matrix"
    ]

    print("Baseline definition, training, and validation")
    # Baseline
    dt_hpar = {
        "criterion": "entropy",
        "max_depth": 20,
        "max_leaf_nodes": 1000
    }
    baseline = DecisionTree(processor, embedder, **dt_hpar)

    baseline.train(train_df["text"], train_df["label"])
    predi = baseline.predict(val_df["text"])
    evalu = baseline.evaluate(predi, val_df["label"], metrics)

    for metric_name, ev in zip(metric_names, evalu):
        print(metric_name + ":")
        print(ev)

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
