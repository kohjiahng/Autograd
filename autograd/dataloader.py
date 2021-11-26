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
            np.random.seed(seed)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)
    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return [self.dataset[i] for i in idx]
        else:
            return self.dataset[idx]
    def __iter__(self):
        _order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(_order)
        _order = np.array_split(_order, len(self))
        return DataLoaderIterator(self, _order)

class DataLoaderIterator:
    def __init__(self, dataloader, order):
        self._dataloader = dataloader
        self._order = order
        self._index = 0
    def __next__(self):
        if self._index == len(self._dataloader):
            raise StopIteration
        data = self._dataloader[self._order[self._index]]
        self._index += 1
        if len(data) > 0 and isinstance(data[0], Iterable):
            return tuple(
                # np.vstack(field) if len(field) > 0 and isinstance(field[0],np.ndarray)\
                np.array(field) for field in zip(*data)
            )
        else:
            return np.array(data)
        


