import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TreeSATDataset(Dataset):
    def __init__(self, file: str) -> None:
        df = pd.read_csv(file, header=None)
        lines = torch.tensor(df.values, dtype=torch.float32)
        self.x = lines[:, :-1]
        self.y = lines[:, -1]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index].unsqueeze(0)


if __name__ == "__main__":
    dataset = TreeSATDataset("./data/datasets/tres_sat_dataset_20_10.csv")
    print(len(dataset))
    dataloader = DataLoader(dataset, 64, shuffle=True)
    input, output = next(iter(dataloader))
    print(input[0].size(), output[0].unsqueeze(0).size())
