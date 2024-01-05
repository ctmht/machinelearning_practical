import multiprocessing

import numpy as np
from gensim.models import Word2Vec


class Embedder:
    """ Class handling word vector embeddings """

    def __init__(self, folder_path: str):
        """
        Create an embedder containing an empty (untrained) Word2Vec model
        at the specified folder path
        Args:
            folder_path: where the Word2Vec model will be saved and loaded
        """
        self.path: str = folder_path + "w2v/trained_w2v.model"

        self.embed_size: int = 100
        self.w2v_model = Word2Vec(
            min_count = 1,
            window = 3,
            vector_size = self.embed_size,
            workers = multiprocessing.cpu_count() - 1
        )
        self._trained: bool = False

    def create_embeddings(self, data: list[list[str]] | list[str]) -> None:
        """
        (Re)trains the Word2Vec model to contain embeddings of all words found
        in the data
        Args:
            data: list of tweets or a single tweet whose contents should be
                added to the available trained embeddings of the model
        """
        # Cover both data parameter formats
        if not isinstance(data[0], list):
            data = [data]

        # Add data to vocabulary
        self.w2v_model.build_vocab(data, update = self._trained)

        # Create embeddings
        self.w2v_model.train(
            data,
            total_examples = 1 if self._trained
                               else self.w2v_model.corpus_count,
            epochs = 20
        )
        self._trained = True

    def __getitem__(self, word: str) -> np.array:
        """
        Get the vector embedding of a word, if necessary add it to the model
        Args:
            word: the word
        Return:
            : the embedding
        """
        if word not in self.w2v_model.wv.key_to_index.keys():
            self.create_embeddings([word])

        return self.w2v_model.wv[word]

    def embed(self, prc_tweet: list[str]) -> list[np.array]:
        """
        Get all embeddings for words in a list
        Args:
            prc_tweet: list of (preprocessed) words
        Return:
            : list of vector embeddings
        """
        return [self[word] for word in prc_tweet]

    def uniform_tweet_embedding(
            self,
            embeds_list: list[str],
            size: int
        ) -> np.array:
        uniform_tweet_emb = np.block(self.embed(embeds_list))
        uniform_tweet_emb.resize((size * self.embed_size,))

        return uniform_tweet_emb

    def save_model(self):
        assert self._trained, "Model not trained yet, not saving"
        self.w2v_model.save(self.path)

    def load_model(self):
        self.w2v_model = Word2Vec.load(self.path)
        self._trained = True