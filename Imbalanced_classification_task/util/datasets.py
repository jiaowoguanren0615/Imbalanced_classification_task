import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, item):
        x_sample = self.X[item]
        y_sample = self.y[item]
        return x_sample, y_sample

    def __len__(self):
        return len(self.y)