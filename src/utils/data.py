"""Utility functions for handling datasets and dataloaders."""

import math
from src.tensor import Tensor
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

class Dataset():
    def __init__(self, data, labels=None, to_tensor:bool = False, target_to_tensor:bool = False, device = "cpu"):
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

        # Device handling
        if device == "cuda" and not CUDA_AVAILABLE:
            raise ValueError("Cupy is not available. Please install cupy to use CUDA.")
        elif device not in ["cpu", "cuda"]:
            raise ValueError("Device must be either 'cpu' or 'cuda'.")
        self.device = device

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

        # Convertir a tensor si es necesario
        if self.to_tensor:
            data = Tensor(data, device=self.device)

        if self.labels is not None:
            lbl = self.labels[index]

            # Convertir a tensor si es necesario
            if self.target_to_tensor:
                    lbl = Tensor(lbl, device=self.device)

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
    def __init__(self, dataset:Dataset,  batch_size:int = 8, shuffle=True, device="cpu"):
        """A simple dataloader class.

        Args:
            dataset (Dataset): The dataset to load data from.
            batch_size (int, optional): The number of samples per batch. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
        """
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = device

    def __iter__(self):
        """Starts an iteration over the dataloader.

        Returns:
            (Dataloader): The dataloader instance.
        """
        xp = self.xp


        if self.shuffle:
            self.indices = xp.random.permutation(len(self.dataset))
        else:
            self.indices = xp.arange(len(self.dataset))

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
        
        # Convertir índices a numpy si es necesario para indexación
        if self.device == "cuda" and CUDA_AVAILABLE:
            batch_indices = cp.asnumpy(batch_indices)

        batch = [self.dataset._getitem(int(i)) for i in batch_indices]
        data_batch, label_batch = zip(*batch)

        self.current += self.batch_size

        data_batch = Tensor(data_batch, requires_grad=True, device=self.device) if self.dataset.to_tensor else data_batch
        label_batch = Tensor(label_batch, device=self.device) if self.dataset.target_to_tensor else label_batch

        return data_batch, label_batch


    def __len__(self):
        """Returns the number of batches in the dataloader.

        Returns:
            (int): The number of batches.
        """
        # return int(len(self.dataset) / self.batch_size)
        return math.ceil(len(self.dataset) / self.batch_size)
    
    @property
    def xp(self):
        return cp if self.device == "cuda" and CUDA_AVAILABLE else np




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
    
    # Detectar si los datos están en GPU o CPU
    xp = cp if CUDA_AVAILABLE and cp is not None and isinstance(dataset.data, cp.ndarray) else np
    
    index_list = xp.random.permutation(len(dataset))

    subsets = []
    start = 0
    for length in lengths:
        part = index_list[start: start + length]
        
        # Convertir índices a numpy para indexación segura
        if xp == cp:
            part = cp.asnumpy(part)
            
        batch = [dataset._getitem(int(i)) for i in part]
        
        data, labels = zip(*batch)

        subsets.append(Dataset(data, labels=labels, to_tensor=dataset.to_tensor, target_to_tensor=dataset.target_to_tensor, device=dataset.device))

        start += length
    
    return subsets
