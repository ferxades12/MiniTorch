from src.tensor import Tensor
import numpy as np

class Dataset():
    def __init__(self, data, labels=None, to_tensor:bool = False, target_to_tensor:bool = False):
        if labels is not None and len(labels) != len(data):
            raise ValueError("Data and labels have not the same size")


        self.data = data
        self.labels = labels
        self.to_tensor = to_tensor
        self.target_to_tensor = target_to_tensor

    def __getitem__(self, index):
        data = self.data[index]
        data = Tensor(data) if self.to_tensor else data    

        if self.labels is not None:
            lbl = self.labels[index]
            lbl = Tensor(lbl) if self.target_to_tensor else lbl

            return (data, lbl)

        return (data, None)
    
    def _getitem(self, index):
        data = self.data[index]

        if self.labels is not None:
            lbl = self.labels[index]

            return (data, lbl)

        return (data, None)
    
    def __len__(self):
        return len(self.data)

class Dataloader:
    def __init__(self, dataset:Dataset,  batch_size:int = 8, shuffle=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = np.arange(len(self.dataset))

        self.current = 0

        return self
    
    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.current: self.current + self.batch_size]

        batch = [self.dataset._getitem(i) for i in batch_indices]
        data_batch, label_batch = zip(*batch)

        self.current += self.batch_size

        return Tensor(data_batch), Tensor(label_batch)


    def __len__(self):
        int(len(self.dataset) / self.batch_size)




def random_split(dataset:Dataset, lengths):
    if sum(lengths) == 1:
        lengths = [int(l * len(dataset)) for l in lengths]

        for i in range(sum(lengths), len(dataset)):
            lengths[i % len(lengths)] += 1
        
    index_list = np.random.permutation(len(dataset))


    subsets = []
    start = 0
    for length in lengths:
        part = index_list[start: start + length]

        data = [dataset.data[j] for j in part]
        labels = None

        if dataset.labels is not None:
            labels = [dataset.labels[j] for j in part]

        subsets.append(Dataset(data, labels=labels, to_tensor=dataset.to_tensor, target_to_tensor=dataset.target_to_tensor))

        start +=length
    
    return subsets
        
        
        

    def __len__(self):
        return len(self.data)