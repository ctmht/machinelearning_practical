from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout
from keras.layers import LSTM as keras_LSTM

from src.model_interface import ModelInterface
from src.processor import Processor
from src.embedder import Embedder


class LSTM(ModelInterface):
    def __init__(self, processor: Processor, embedder: Embedder, **kwargs):
        super().__init__(processor, embedder)

        self.model = Sequential()

        self.build_model(max_sequence_length = 20) #????

    def build_model(self, max_sequence_length):
        # Retrieve the embeddings from the Word2Vec model
        self.embedder.load_model()
        embedding_matrix = self.embedder.w2v_model.wv.vectors
        vocab_size, embedding_dim = embedding_matrix.shape

        self.model.add(
            Embedding(
                input_dim = vocab_size,
                output_dim = embedding_dim,
                weights = [embedding_matrix],
                input_length = max_sequence_length,
                trainable = False
            )
        )
        self.model.add(
            keras_LSTM(
                units = 50,
                return_sequences = False
            )
        )
        self.model.add(
            Dense(
                1,
                activation = "softmax"
            )
        )

        # Compile model
        self.model.compile(
            optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
