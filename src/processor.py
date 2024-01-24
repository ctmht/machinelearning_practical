from collections.abc import Iterator

import wordsegment

from src.tweet import Tweet
from src.augmenter import Augmenter
from src.util import load


class Processor:
    """ Class handling whole file preprocessing + saving and loading data """

    targets = ["train", "val", "test"]

    def __init__(self, folder_path: str):
        """
        Create a preprocessor object with access to the unprocessed data
        for further processing
        Args:
            folder_path: relative path to the folder containing the target
                data. It is assumed that the filenames are *_text.txt and
                *_labels.txt, respectively, within this folder
        """
        # Define paths
        self.tf = folder_path
        self.pf = folder_path + "prc_data/"
        self.af = folder_path + "aug_data/"

        wordsegment.load()

    def get_data(self, from_files: bool=True, augmented: bool=False) -> dict:
        """
        Get preprocessed data formatted as a dictionary
        Return:
            : dictionary with data type (train, val, test), data label, and
                data text (preprocessed list of lemmas)
        """
        if from_files:
            return self._get_data_from_files(augmented)
        else:
            print("Getting files from preprocessing", flush = True)
            data = self._get_data_from_preprocessing(augmented)

            return data

    def _get_data_from_files(self, augmented: bool) -> dict:
        """ Get already preprocessed data from files """
        data = {"type": [], "label": [], "text": []}
        for target in Processor.targets:
            path = self.af if augmented and target == "train" else self.pf
            proctype = "_aug" if augmented and target == "train" else "_prc"
            tfile, lfile = load(path, target, 'r', proctype)

            for tw, lab in zip(tfile.readlines(), lfile.readlines()):
                data["type"].append(target)
                data["label"].append(int(lab))
                data["text"].append(tw[:-1].split(' '))

        return data

    def _get_data_from_preprocessing(self, augmented: bool) -> dict:
        """ Preprocess data and return results """
        data = {"type": [], "label": [], "text": []}

        for yld_res in self._preprocess_files(augmented, res_yield = True):
            # Add data type to entries of this step
            data["type"].extend([yld_res["target"]] * len(yld_res["data"]))

            # Add label and text to entries of this step
            for entry in yld_res["data"]:
                data["label"].append(entry[1])
                data["text"].append(entry[0])

        return data

    def _preprocess_files(
            self,
            augment: bool,
            res_yield: bool=False
        ) -> Iterator[dict]:
        """
        Applies preprocessing to files and stores into new files
        Args:
            res_yield: whether to yield what this call preprocesses
        """
        for target in Processor.targets:
            # Load the new files
            print("Preprocessing " + target)

            if not res_yield:
                self._preprocess_textlab_pair(target, False)
            elif target != "train" or not augment:
                yield {
                    "target": target,
                    "data": self._preprocess_textlab_pair(target, True)
                }
            else:
                # self._preprocess_textlab_pair(target, False)
                yield {
                    "target": target,
                    "data": self._augment_textlab_pair(True)
                }

    def _preprocess_textlab_pair(
            self,
            target: str,
            res_return: bool=False
        ) -> list[(list[str], int)] | None:
        """
        Loop through entries in the loaded files and apply preprocessing
        Args:
            res_return: whether to also return the saved processed data
        """
        self._load_target(target)

        if res_return:
            prc_data: [([str], int)] = []

        for tweet, label in zip(self.target_tfile, self.target_lfile):
            # Get preprocessed original tweet
            prc_tweet: list[str] = self.preprocess(tweet)

            self.prc_target_tfile.write(" ".join(prc_tweet) + '\n')
            self.prc_target_lfile.write(label)

            if res_return:
                prc_data.append((prc_tweet, int(label)))

        if res_return:
            return prc_data

    @staticmethod
    def preprocess(tweet: str) -> list[str]:
        """ Wrapper for Tweet.preprocess, giving list of processed words """
        return Tweet(tweet).preprocess()

    def _load_target(self, target: str) -> None:
        """ Reloads the target and processed files into class members """
        print("Loading target " + target, flush=True)

        self.target_tfile, self.target_lfile = load(
            self.tf, target, 'r'
        )
        self.prc_target_tfile, self.prc_target_lfile = load(
            self.pf, target, 'w', "_prc"
        )

    def _augment_textlab_pair(self, read_results: bool=False)\
            -> list[([str], int)] | None:
        augmenter = Augmenter(self.tf)
        augmenter.augment()

        if read_results:
            aug_data: list[(list[str], int)] = augmenter.read_augdata()
            return aug_data