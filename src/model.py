from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM as keras_LSTM


class LSTM:
    def __init__(self, embedding_matrix, max_sequence_length, num_classes):
        self.max_sequence_length = max_sequence_length
        self.model = Sequential()
        self.build_model(embedding_matrix, num_classes)

    def build_model(self, embedding_matrix, num_classes):

        # Retrieve the weights from the Word2Vec model
        vocab_size, embedding_dim = embedding_matrix.shape

        # Build model
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                 input_length=self.max_sequence_length, trainable=False))
        self.model.add(keras_LSTM(units=100, return_sequences=False))
        self.model.add(Dense(num_classes, activation='softmax'))

        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=1, batch_size=None, validation_split=0, class_weight=None):
        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split,
                       class_weight=class_weight)

    def predict(self, X):
        ret = self.model.predict(X)
        return ret

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)
