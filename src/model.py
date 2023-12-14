from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout
from keras.layers import LSTM as keras_LSTM


class LSTM():
    def __init__(self, w2v_model, max_sequence_length, num_classes):
        self.model = Sequential()
        self.build_model(w2v_model, max_sequence_length, num_classes)

    def build_model(self, w2v_model, max_sequence_length, num_classes):

        # Retrieve the weights from the Word2Vec model
        embedding_matrix = w2v_model.wv.vectors
        vocab_size, embedding_dim = embedding_matrix.shape

        # Build model
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                 input_length=max_sequence_length, trainable=False))
        self.model.add(keras_LSTM(units=50, return_sequences=False))
        self.model.add(Dense(1, activation='softmax'))

        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=1):
        self.model.fit(X, y, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)
