import pandas as pd

from src.processor import Processor


def main():
    # LOAD DATA
    load_from_files = True      # load from files if True, preprocess if False

    processor = Processor("../data/")
    data = processor.get_data(from_files = load_from_files)
    df = pd.DataFrame(data)
    print(df)

    # GET VECTOR EMBEDDINGS FOR ALL WORDS IN ALL TEXT FILES
    w2v_model = processor.get_word2vec_model() # not implemented yet
    # TODO (later): allow for dataset extensions (e.g. custom test tweets) in
    # TODO: both general processing and words known by the w2v_model

    # TODO: General 'Model' class with utils as the word2vec usage above

    # TODO: Baseline and LSTM classes with docs, inheriting from Model

    # TODO: Baseline/LSTM will get the w2v_model and use it directly to
    # TODO: get the inputs from the words themselves (memory efficiency - so
    # TODO: not loading all wordvecs into the df directly bc it's redundant)

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
    # baseline.train(create_tweet_embeddings(w2v_model, train_tweets), train_labels)
    #
    # print("lstm")
    # model = LSTM(w2v_model, 50, 20)
    # print(model.model.summary())
    # model.train(create_tweet_encodings(w2v_model, train_tweets), train_labels)
    #
    # # Evaluate both models
    # print("evaluate")
    # baseline_pred = baseline.predict(create_tweet_embeddings(w2v_model, val_tweets))
    # loss, accuracy = model.evaluate(create_tweet_encodings(w2v_model, val_tweets), val_labels)
    # print('Accuracy baseline: \t%f' % (np.sum(baseline_pred == val_labels) / val_labels.size * 100))
    # print('Accuracy model: \t%f' % (accuracy * 100))
    # print('Random guess: \t%f' % (np.max(np.unique(val_labels, return_counts=True)[1]) / val_labels.size * 100))


if __name__ == '__main__':
    main()
