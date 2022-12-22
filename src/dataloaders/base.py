from .negative_samplers import negative_sampler_factory

from abc import *


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(
        self, dataset, data_type, test_negative_sampler_code, test_negative_sample_size
    ):
        save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset["train"]
        self.val = dataset["val"]
        self.test = dataset["test"]
        self.test_negative_sampler_code = test_negative_sampler_code
        if "user_count" in dataset.keys():
            self.user_count = dataset["user_count"]
            print("num of users", self.user_count)
        self.item_count = dataset["item_count"]

        if test_negative_sampler_code in ["random", "popualar"]:
            test_negative_sampler = negative_sampler_factory(
                test_negative_sampler_code,
                self.train,
                self.val,
                self.test,
                self.user_count,
                self.item_count,
                test_negative_sample_size,
                save_folder,
            )

            self.test_negative_samples = test_negative_sampler.get_negative_samples()
            self.val_negative_samples = test_negative_sampler.get_negative_samples()
        else:
            if data_type == "session":
                self.val_negative_samples = dataset["val_sample"]
                self.test_negative_samples = dataset["test_sample"]
            else:
                self.val_negative_samples = dataset["sample_seq"]
                self.test_negative_samples = dataset["sample_seq"]

    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_valid_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass
