from .base import AbstractDataloader

import torch
import numpy as np
import torch.utils.data as data_utils


class DualRecDataloader(AbstractDataloader):
    def __init__(
        self,
        dataset,
        data_type,
        seg_len,
        num_train_seg,
        num_test_seg,
        pred_prob,
        num_workers,
        test_negative_sampler_code,
        test_negative_sample_size,
        train_batch_size,
        val_batch_size,
        test_batch_size,
    ):
        super().__init__(
            dataset, data_type, test_negative_sampler_code, test_negative_sample_size
        )
        self.seg_len = seg_len
        self.data_type = data_type
        self.num_train_seg = num_train_seg
        self.num_test_seg = num_test_seg
        self.pred_prob = pred_prob
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def _get_train_dataset(self):
        dataset = DualRecTrainDataset(
            self.train,
            self.data_type,
            self.seg_len,
            self.num_train_seg,
            self.pred_prob,
            self.item_count,
        )
        return dataset

    def get_valid_loader(self):
        return self._get_eval_loader(mode="val")

    def get_test_loader(self):
        return self._get_eval_loader(mode="test")

    def _get_eval_loader(self, mode):
        batch_size = self.val_batch_size if mode == "val" else self.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == "val" else self.test
        if self.data_type == "seq":
            dataset = DualRecEvalDataset(
                self.train,
                self.data_type,
                answers,
                self.seg_len,
                self.num_test_seg,
                self.val_negative_samples
                if mode == "val"
                else self.test_negative_samples,
                mode,
            )
        else:
            dataset = DualRecEvalDataset(
                self.val if mode == "val" else self.test,
                self.data_type,
                answers,
                self.seg_len,
                self.num_test_seg,
                self.val_negative_samples
                if mode == "val"
                else self.test_negative_samples,
                mode,
            )
        return dataset


class DualRecTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, data_type, seg_len, num_seg, pred_prob, num_items):
        self.u2seq = u2seq
        self.data_type = data_type
        self.users = (
            sorted(self.u2seq.keys())
            if self.data_type == "seq"
            else torch.arange(len(self.u2seq))
        )
        self.seg_len = seg_len + 1
        self.num_seg = num_seg
        self.max_len = seg_len * num_seg
        self.pred_prob = pred_prob
        self.max_pred = int(seg_len * pred_prob)
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        if self.data_type == "seq":
            seq = self._getseq(user)[:-1]
        else:
            seq = self._getseq(user)
        if len(seq) > self.max_len:
            begin_idx = np.random.randint(0, len(seq) - self.max_len + 1)
            seq = seq[begin_idx : begin_idx + self.max_len]

        ret = {"input_ids": []}

        for i in range(self.num_seg):
            seg = seq[
                max(0, len(seq) - (i + 1) * self.seg_len) : len(seq) - i * self.seg_len
            ]
            # padding
            padding_len = self.seg_len - len(seg)
            seg = [0] * padding_len + seg

            ret["input_ids"].insert(0, torch.LongTensor(seg))

        # stack different segment together
        ret = {k: torch.stack(v) for k, v in ret.items()}

        return ret

    def _getseq(self, user):
        return self.u2seq[user]

    def neg_sample(self, item_set, size):
        negs = []
        while len(negs) < size:
            item = torch.randint(1, self.num_items, (1,))[0]
            while item in item_set:
                item = torch.randint(1, self.num_items, (1,))[0]
            negs.append(item)
        return negs


class DualRecEvalDataset(data_utils.Dataset):
    def __init__(
        self, u2seq, data_type, u2answer, seg_len, num_seg, negative_samples, mode
    ):
        self.u2seq = u2seq
        self.data_type = data_type
        self.users = (
            sorted(self.u2seq.keys())
            if self.data_type == "seq"
            else torch.arange(len(self.u2seq))
        )
        self.u2answer = u2answer
        self.seg_len = seg_len
        self.num_seg = num_seg
        self.max_len = seg_len * num_seg
        self.negative_samples = negative_samples
        self.mode = mode

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        if self.mode == "val" or self.data_type == "session":
            seq = self.u2seq[user][:-1]
        else:
            seq = self.u2seq[user]
        if self.data_type == "seq":
            answer = self.u2answer[user]
        else:
            answer = self.u2answer[user][-1:]

        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        # Add dummy token at the end (no attention on this one)
        seq = seq[-self.max_len :]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        ret = {
            "input_ids": [],
            "labels": [torch.LongTensor(labels)],
            "candidates": [torch.LongTensor(candidates)],
        }

        for i in range(self.num_seg):
            seg = seq[i * self.seg_len : (i + 1) * self.seg_len]

            ret["input_ids"].append(torch.LongTensor(seg))
        # stack different segment together
        ret = {k: torch.stack(v) for k, v in ret.items()}
        return ret
