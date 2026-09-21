########################### Dataset class ##########################


import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {}

        for key, value in self.encodings.items():
            item[key] = torch.tensor(value[idx])

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)