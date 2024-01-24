import re
import spacy
from nltk.corpus import stopwords

import wordsegment


class Tweet:
    """ Container class for a tweet handling sentence-level preprocessing """

    nlp = spacy.load("en_core_web_trf", disable=['parser', 'ner'])
    stop_words = set(stopwords.words("english"))

    def __init__(self, tweet: str):
        """
        Creates a container for a tweet
        Args:
            tweet: the string of a tweet
        """
        self.tweet: str = tweet
        self.prc_nlp = None
        self.prc_tweet = None

    def preprocess(self) -> list[str]:
        """
        Preprocesses the tweet by removing cluttering characters (punctuation,
        digits, extra spaces), lemmatizing, (and removing stopwords)
        Return:
            self.prc_tweet: list of lemmas following these preprocessing steps
        """
        # Split hashtags
        prc = Tweet.split_hashtags(self.tweet)

        # Clean non-words
        prc = Tweet.clean_nonwords(prc)

        # Lemmatize string (not removing stopwords now)
        self.prc_nlp = Tweet.nlp(prc)
        lemmas: [str] = [token.lemma_.lower() for token in self.prc_nlp]
        self.prc_tweet = lemmas # Tweet.remove_stopwords(lemmas) for RS

        return self.prc_tweet

    @staticmethod
    def clean_nonwords(text: str) -> str:
        """
        Method for removing punctuation, digits, and extra whitespace from a
        given text string
        Args:
            text: string to be processed
        Return:
            result: lowercase string after cleaning non-word characters
        """
        result = text.lower()

        # Replace punctuation and digits with space
        result = re.sub(
            r"[…!\"#$%&\'()*+,-./:;<=>? ️@\[\]^_`{|}~\d]",
            " ",
            result
        )

        # Limit whitespaces to one space
        result = re.sub(r"(\s+)", " ", result).strip()

        return result

    @staticmethod
    def remove_stopwords(lemmas: [str]) -> [str]:
        """
        Removes stopwords from a list of lemmas
        Args:
            lemmas: list of lemmas
        Return:
            : list of lemmas with no stopwords
        """
        return [
            lemma for lemma in lemmas if lemma not in Tweet.stop_words
        ]

    @staticmethod
    def split_hashtags(text: str) -> str:
        pattern = r"#(\w+)"
        words = text.split()
        for index, word in enumerate(words):
            if re.match(pattern, word):
                split_words = Tweet.split_word(word)

                text = text.replace(word, split_words)
        return text

    @staticmethod
    def split_word(word: str) -> str:
        split_words = wordsegment.segment(word)

        return " ".join(str(word) for word in split_words)