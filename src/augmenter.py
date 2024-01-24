import transformers
import nlpaug.augmenter.word as naw
import numpy as np

from src.util import load


class Augmenter:
    aug_bert = naw.ContextualWordEmbsAug(
        model_type = "bert", aug_min = 2, aug_max = 5
    )
    aug_wnet = naw.SynonymAug(
        aug_src = "wordnet", aug_min = 2, aug_max = 5
    )

    def __init__(self, folder_path: str, num_classes: int=20):
        print("\tCreating augmenter")

        self.pf = folder_path + "prc_data/"
        self.af = folder_path + "aug_data/"

        self.frequencies: list[int] = num_classes * [0]
        self.frequencies_post: list[int] = num_classes * [0]
        self._get_trainfreqs()

    def _get_trainfreqs(self, post_augment: bool=False):
        if post_augment:
            text, labs = load(self.af, "train", 'r', "_aug")
            target = self.frequencies_post
        else:
            text, labs = load(self.pf, "train", 'r', "_prc")
            target = self.frequencies

        print("\tGathering train class frequencies from ", labs)

        for label in labs:
            label = int(label)
            target[label] += 1

        print("\tFrequencies: ", target)

    def _get_augments(self, tweet: str, aug_ratio: int):
        return np.concatenate([
            self.aug_bert.augment(tweet, n = aug_ratio // 2),
            self.aug_wnet.augment(tweet, n = aug_ratio // 2)
        ])[0 : (aug_ratio - 1)]

    def augment(self):
        print("\n\tAugmenting")

        prc_text, prc_labs = load(self.pf, "train", 'r', "_prc")
        aug_text, aug_labs = load(self.af, "train", 'w', "_aug")

        max_freq = max(self.frequencies)
        max_ind = self.frequencies.index(max_freq)

        for tweet, label in zip(prc_text, prc_labs):
            # Keep original tweet
            aug_text.write(tweet)
            aug_labs.write(label)

            # Augment enough to match highest pre-aug class
            this_label = int(label)
            this_freq = self.frequencies[this_label]

            if this_freq != max_freq:
                for aug_tweet in self._get_augments(
                    tweet, int(max_freq / this_freq)
                ):
                    aug_tweet += "\n" if aug_tweet[-1] else ""
                    aug_text.write(aug_tweet)
                    aug_labs.write(label)

        self._get_trainfreqs(post_augment = True)

    def read_augdata(self) -> list[(list[str], int)]:
        aud: list[(list[str], int)] = []

        rft, rfl = load(self.af, "train", 'r', "_aug")

        for aug_tweet, label in zip(rft, rfl):
            aud.append((aug_tweet.split(' '), int(label)))

        return aud