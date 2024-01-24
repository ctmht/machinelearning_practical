from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM as keras_LSTM
from saving_loading import save, load


class LSTM:
    def __init__(self, embedding_matrix, max_sequence_length, num_classes, units):
        self.max_sequence_length = max_sequence_length
        self.model = Sequential()
        self.build_model(embedding_matrix, num_classes, units)

    def build_model(self, embedding_matrix, num_classes, units):

        # Retrieve the weights from the Word2Vec model
        vocab_size, embedding_dim = embedding_matrix.shape

        # Build model
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                 input_length=self.max_sequence_length, trainable=False))
        self.model.add(keras_LSTM(units=units, return_sequences=False))
        self.model.add(Dense(num_classes, activation='softmax'))

        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=1, batch_size=None):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        ret = self.model.predict(X)
        return ret

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)


def initialize_and_train_lstm(embedding_matrix, max_tweet_len, units, epochs, batch_size, name,
                              train_tweet_padded_embeddings, train_labels_one_hot,
                              save_data=False, load_data=False):
    print("LSTM")
    if not load_data:
        model = LSTM(embedding_matrix, max_tweet_len, num_classes=20, units=units)
        print(model.model.summary())
        model.train(train_tweet_padded_embeddings, train_labels_one_hot, epochs=epochs, batch_size=batch_size)
    else:
        model = load('../models/' + name + '.pkl')

    if save_data:
        save(model, '../models/' + name + '.pkl')
    return model
