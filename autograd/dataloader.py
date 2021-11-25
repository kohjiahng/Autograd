import random
import numpy as np
import logging
from collections.abc import Iterable 
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
            random.shuffle(_order)
        return DataLoaderIterator(self, _order)

class DataLoaderIterator:
    def __init__(self, dataloader, order):
        self._dataloader = dataloader
        self._order = order
        self._index = 0
    def __next__(self):
        if self._index >= len(self._dataloader.dataset):
            raise StopIteration

        left = self._index
        right = min(len(self._dataloader.dataset), self._index + self._dataloader.batch_size)
        self._index += self._dataloader.batch_size
        self._index = right
        data = [self._dataloader.dataset[self._order[i]] for i in range(left, right)]
        if len(data) > 0 and isinstance(data[0], Iterable):
            return tuple(
                # np.vstack(field) if len(field) > 0 and isinstance(field[0],np.ndarray)\
                np.array(field) for field in zip(*data)
            )
        else:
            return np.array(data)
        


