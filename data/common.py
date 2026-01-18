import torch
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


class PairedDaraset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        x = (self.data1[idx], self.data2[idx], self.labels[idx])
        return x