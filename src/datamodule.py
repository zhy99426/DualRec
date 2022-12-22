import pytorch_lightning as pl
from .datasets import AbstractDataset
from .dataloaders import DualRecDataloader


class DualRecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_code: str = None,
        data_type: str = None,
        data_file_eval: str = None,
        data_file_test: str = None,
        seg_len: int = None,
        num_train_seg: int = None,
        num_test_seg: int = None,
        pred_prob: float = None,
        num_workers: int = None,
        test_negative_sampler_code: str = None,
        test_negative_sample_size: int = None,
        train_batch_size: int = None,
        val_batch_size: int = None,
        test_batch_size: int = None,
    ):
        super().__init__()
        self.dataset_code = dataset_code
        self.data_type = data_type
        self.data_file_eval = data_file_eval
        self.data_file_test = data_file_test
        self.seg_len = seg_len
        self.num_train_seg = num_train_seg
        self.num_test_seg = num_test_seg
        self.pred_prob = pred_prob
        self.num_workers = num_workers
        self.test_negative_sampler_code = test_negative_sampler_code
        self.test_negative_sample_size = test_negative_sample_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        AbstractDataset(self.dataset_code, self.data_file_eval, self.data_file_test)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.dataset = AbstractDataset(
            self.dataset_code, self.data_file_eval, self.data_file_test
        )
        self.dataloader = DualRecDataloader(
            self.dataset,
            self.data_type,
            self.seg_len,
            self.num_train_seg,
            self.num_test_seg,
            self.pred_prob,
            self.num_workers,
            self.test_negative_sampler_code,
            self.test_negative_sample_size,
            self.train_batch_size,
            self.val_batch_size,
            self.test_batch_size,
        )

    def train_dataloader(self):
        return self.dataloader.get_train_loader()

    def val_dataloader(self):
        return self.dataloader.get_valid_loader()

    def test_dataloader(self):
        return self.dataloader.get_test_loader()
