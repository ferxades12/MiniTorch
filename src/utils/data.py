"""Utility functions for handling datasets and dataloaders."""

from src.tensor import Tensor
import numpy as np

class Dataset():
    def __init__(self, data, labels=None, to_tensor:bool = False, target_to_tensor:bool = False):
        """A simple dataset class.

        Args:
            data (Any): The data of the dataset
            labels (Any, optional): The labels of the dataset. Defaults to None.
            to_tensor (bool, optional): Whether to convert the data to tensors. Defaults to False.
            target_to_tensor (bool, optional): Whether to convert the labels to tensors. Defaults to False.

        Raises:
            ValueError: If the lengths of data and labels do not match.
        """
        if labels is not None and len(labels) != len(data):
            raise ValueError("Data and labels have not the same size")


        self.data = data
        self.labels = labels
        self.to_tensor = to_tensor
        self.target_to_tensor = target_to_tensor

    def __getitem__(self, index):
        """Retrieves an item from the dataset.

        Args:
            index (Union[int, slice]): The index or slice to retrieve from the dataset.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The data and labels for the given index.
        """
        if isinstance(index, slice):
            idx = range(*index.indices(len(self)))
            return [self[i] for i in idx]

        data = self.data[index]
        data = Tensor(data) if self.to_tensor else data    

        if self.labels is not None:
            lbl = self.labels[index]
            lbl = Tensor(lbl) if self.target_to_tensor else lbl

            return (data, lbl)

        return (data, None)
    
    def _getitem(self, index):
        """Private method to return the item without the transformation.

        Args:
            index (int): The index to retrieve from the dataset.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The data and labels for the given index.
        """
        data = self.data[index]

        if self.labels is not None:
            lbl = self.labels[index]

            return (data, lbl)

        return (data, None)
    
    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            (int): The number of samples in the dataset.
        """
        return len(self.data)

class Dataloader:
    def __init__(self, dataset:Dataset,  batch_size:int = 8, shuffle=True):
        """A simple dataloader class.

        Args:
            dataset (Dataset): The dataset to load data from.
            batch_size (int, optional): The number of samples per batch. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
        """
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        """Starts an iteration over the dataloader.

        Returns:
            (Dataloader): The dataloader instance.
        """
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = np.arange(len(self.dataset))

        self.current = 0

        return self
    
    def __next__(self):
        """Retrieves the next batch of data from the dataloader.

        Raises:
            StopIteration: If there are no more batches to retrieve.
        Returns:
            Tuple[Tensor, Optional[Tensor]]: The next batch of data and labels.
        """
        if self.current >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.current: self.current + self.batch_size]

        batch = [self.dataset._getitem(i) for i in batch_indices]
        data_batch, label_batch = zip(*batch)

        self.current += self.batch_size

        data_batch = Tensor(data_batch, requires_grad=True) if self.dataset.to_tensor else data_batch
        label_batch = Tensor(label_batch) if self.dataset.target_to_tensor else label_batch

        return data_batch, label_batch


    def __len__(self):
        """Returns the number of batches in the dataloader.

        Returns:
            (int): The number of batches.
        """
        return int(len(self.dataset) / self.batch_size)




def random_split(dataset:Dataset, lengths):
    """Splits a dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset (Dataset): The dataset to split.
        lengths (list): A list of lengths for the splits.
    Returns:
        (list): A list of Dataset objects representing the splits.
    """
    if sum(lengths) == 1:
        lengths = [int(l * len(dataset)) for l in lengths]

        for i in range(sum(lengths), len(dataset)):
            lengths[i % len(lengths)] += 1
        
    index_list = np.random.permutation(len(dataset))


    subsets = []
    start = 0
    for length in lengths:
        part = index_list[start: start + length]

        batch = [dataset._getitem(i) for i in part]
        data, labels = zip(*batch)

        subsets.append(Dataset(data, labels=labels, to_tensor=dataset.to_tensor, target_to_tensor=dataset.target_to_tensor))

        start +=length
    
    return subsets
