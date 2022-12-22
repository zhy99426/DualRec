RAW_DATASET_ROOT_FOLDER = "Data"

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import os
from abc import *
from pathlib import Path
import pickle
from torch.utils.data import RandomSampler, SequentialSampler
from .utils import *

sequential_data_list = ["Beauty", "Sports_and_Outdoors", "Toys_and_Games", "Yelp"]


def get_seq_dic(
    data_name=None, data_dir=None, data_file_eval=None, data_file_test=None
):

    data_file = os.path.join(data_dir, data_name, data_name + ".txt")
    sample_file = os.path.join(data_dir, data_name, data_name + "_sample.txt")
    user_seq, max_item, num_users, sample_seq = get_user_seqs_and_sample(
        data_file, sample_file
    )
    seq_dic = {
        "user_seq": user_seq,
        "num_users": num_users,
        "sample_seq": sample_seq,
    }

    return seq_dic, max_item


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, data_name=None, data_file_eval=None, data_file_test=None):
        self.split = "leave_one_out"
        self.data_name = data_name
        self.data_file_eval = data_file_eval
        self.data_file_test = data_file_test

    def code(self):
        return self.data_name

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open("rb"))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print("Already preprocessed. Skip preprocessing")
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        seq_dic, max_item = get_seq_dic(
            self.data_name,
            self._get_rawdata_root_path(),
            self.data_file_eval,
            self.data_file_test,
        )
        print("num of items:", max_item)
        if self.data_name in sequential_data_list:
            train, val, test = self.split_df(seq_dic)
            dataset = {
                "train": train,
                "val": val,
                "test": test,
                "sample_seq": seq_dic["sample_seq"],
                "user_count": seq_dic["num_users"],
                "item_count": max_item,
            }
        else:
            dataset = {
                "train": seq_dic["user_seq"],
                "val": seq_dic["user_seq_eval"],
                "test": seq_dic["user_seq_test"],
                "val_sample": seq_dic["sample_seq_eval"],
                "test_sample": seq_dic["sample_seq_test"],
                "item_count": max_item,
            }
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)

    def split_df(self, seq_dic):
        if self.split == "leave_one_out":
            print("Splitting")

            train, val, test = {}, {}, {}
            for user in range(seq_dic["num_users"]):
                items = seq_dic["user_seq"][user]
                # train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
                # after find best hyperparameters on validation set, but traini/valid set together
                train[user], val[user], test[user] = (
                    items[:-1],
                    items[-2:-1],
                    items[-1:],
                )
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath("dataset.pkl")

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath("preprocessed")

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = "{}".format(self.code())
        return preprocessed_root.joinpath(folder_name)
