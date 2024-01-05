import io
from collections.abc import Iterator

from src.tweet import Tweet


class Processor:
    """ Class handling whole file preprocessing, saving and loading data """

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

    def get_data(self, from_files: bool=True) -> dict:
        """
        Get preprocessed data formatted as a dictionary
        Return:
            : dictionary with data type (train, val, test), data label, and
                data text (preprocessed list of lemmas)
        """
        if from_files:
            return self._get_data_from_files()
        else:
            return self._get_data_from_preprocessing()

    def _get_data_from_files(self) -> dict:
        """ Get already preprocessed data from files """
        data = {"type": [], "label": [], "text": []}
        for target in Processor.targets:
            tfile, lfile = self._load(self.pf, target, 'r', "_prc")

            all_text = tfile.readlines()
            all_labs = lfile.readlines()

            for tw, lab in zip(all_text, all_labs):
                data["type"].append(target)
                data["label"].append(int(lab))
                data["text"].append(tw[:-1].split(' '))

        return data

    def _get_data_from_preprocessing(self) -> dict:
        """ Preprocess data and return results """
        data = {"type": [], "label": [], "text": []}
        for yld_res in self._preprocess_files(
                aug = False, res_yield = True
            ):
            # Add data type to entries of this step
            data["type"].extend([yld_res["target"]] * len(yld_res["data"]))

            # Add label and text to entries of this step
            for entry in yld_res["data"]:
                data["label"].append(entry[1])
                data["text"].append(entry[0])

        return data

    def _preprocess_files(
            self,
            aug: bool=False,
            res_yield: bool=False
        ) -> Iterator[dict]:
        """
        Applies preprocessing to files and stores into new files, optionally
        augmenting the training data
        Args:
            aug: whether to augment the training data or not
            res_yield: whether to yield what this call preprocesses
        """
        for target in Processor.targets:
            # Load the new files
            self._load_target(target)

            aug_this = aug if target == "train" else False
            if res_yield:
                yield {
                    "target": target,
                    "data": self._preprocess_textlab_pair(aug_this, True)
                }
            else:
                self._preprocess_textlab_pair(aug_this, False)

    def _preprocess_textlab_pair(
            self,
            augment: bool=False,
            res_return: bool=False
        ) -> list[(list[str], int)] | None:
        """
        Loop through entries in the loaded files and apply preprocessing
        Args:
            augment: whether to augment (can be true for training set only)
            res_return: whether to also return the saved processed data
        """
        if res_return:
            prc_data: [([str], int)] = []

        i: int = 0 # TODO: let this go till the end of the files

        for tweet, label in zip(self.target_tfile, self.target_lfile):
            # Get preprocessed original tweet
            to_write: list[list[str]] = [self.preprocess(tweet)]

            if augment:
                pass # TODO: extend to_write with augmented tweets

            for prc_tweet in to_write:
                # Add original and augmented tweets to processed files
                self.prc_target_tfile.write(" ".join(prc_tweet) + '\n')
                self.prc_target_lfile.write(label)

            if res_return:
                prc_data.extend([(tw, int(label)) for tw in to_write])

            i += 1
            if i == 2:
                break # TODO: let this go till the end of the files

        if res_return:
            return prc_data

    @staticmethod
    def preprocess(tweet: str) -> list[str]:
        """ Wrapper for Tweet.preprocess, giving list of processed words """
        return Tweet(tweet).preprocess()

    def _load_target(self, target: str) -> None:
        """ Reloads the target and processed files into class members """
        self.target_tfile, self.target_lfile = self._load(
            self.tf, target, 'r'
        )
        self.prc_target_tfile, self.prc_target_lfile = self._load(
            self.pf, target, 'w', "_prc"
        )

    @staticmethod
    def _load(ident: str, target: str, intype: str= 'r', suff: str= "")\
            -> (io.FileIO, io.FileIO):
        """ Get text and labels files """
        tpath = ident + target + "_text" + suff + ".txt"
        lpath = ident + target + "_labels" + suff + ".txt"

        tfile = open(tpath, intype, encoding="utf-8")
        lfile = open(lpath, intype, encoding="utf-8")

        return tfile, lfile