import random
import numpy as np
import logging
from math import ceil
class DataLoader:
    def __init__(self, dataset, batch_size = 20, shuffle = True, seed = None):
        '''
        dataset has to be indexed
        '''
        if seed:
            random.seed(seed)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)
    def __getitem__(self, idx):
        return self.dataset[idx]
    def __iter__(self):
        _order = list(range(len(self.dataset)))
        if self.shuffle:
            _order = random.shuffle(_order)
        return DataLoaderIterator(self, _order)

class DataLoaderIterator:
    def __init__(self, dataset, order):
        self._dataset = dataset
        self._order = order
        self._index = 0
    def __next__(self):
        left = self._index
        right = min(len(self._dataset), self._index + self._dataset.batch_size)
        if left == right:
            raise StopIteration
        self._index = right
        data = [self._dataset[i] for i in range(left, right)]
        return tuple(
            # np.vstack(field) if len(field) > 0 and isinstance(field[0],np.ndarray)\
            np.array(field) for field in zip(*data)
        )
        


