from .base import AbstractNegativeSampler

from tqdm import trange
import numpy as np

from collections import Counter


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return "popular"

    def generate_negative_samples(self):
        items = np.arange(self.item_count) + 1
        popularity, total_count = self.get_popularity()
        prob = np.array([popularity[x] / total_count for x in items])
        assert prob.sum() - 1e-9 <= 1.0

        negative_samples = {}
        print("Sampling negative items")
        for user in trange(self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            zeros = np.array(list(seen)) - 1  # items start from 1
            p = prob.copy()
            p[zeros] = 0.0
            p = p / p.sum()
            samples = np.random.choice(items, self.sample_size, replace=False, p=p)

            negative_samples[user] = list(samples)

        return negative_samples

    def get_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        return popularity, sum(popularity.values())
