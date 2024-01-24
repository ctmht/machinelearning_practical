from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from saving_loading import save, load


class HyperLSTM(kt.HyperModel):
    def __init__(self, embedding_matrix, max_sequence_length, num_classes):
        self.embedding_matrix = embedding_matrix
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.batch_size = 0

    def build(self, hp):
        model = Sequential()
        vocab_size, embedding_dim = self.embedding_matrix.shape
        self.batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                            weights=[self.embedding_matrix],
                            input_length=self.max_sequence_length,
                            trainable=False))

        model.add(LSTM(240, return_sequences=False))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


def initialize_and_train_hyper_lstm(embedding_matrix, max_tweet_len,
                                    train_tweet_padded_embeddings, train_labels_one_hot,
                                    save_data=False, load_data=False):
    print("LSTM")
    if not load_data:
        model = HyperLSTM(embedding_matrix, max_tweet_len, num_classes=20)
        tuner = kt.Hyperband(
            model,
            objective="accuracy",
            max_epochs=100,
            factor=3,
            overwrite=True,
            directory="hyper_model",
            project_name="hyperLSTM"
        )

        tuner.search(train_tweet_padded_embeddings, train_labels_one_hot,
                     epochs=50,
                     callbacks=[EarlyStopping(monitor='accuracy', patience=5)])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Retrain the model with the best hyperparameters
        model = model.build(best_hps)
        best_batch_size = best_hps.get('batch_size')
        print(best_batch_size)
        model.fit(train_tweet_padded_embeddings, train_labels_one_hot, epochs=50, batch_size=best_batch_size)

    else:
        model = load('../models/hyper_lstm_model.pkl')

    if save_data:
        save(model, '../models/hyper_lstm_model.pkl')

    return model

# hyperparams: 240 units, 50 epochs, 64 batch_size
