from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TicTacToeDataset(Dataset):
    def __init__(self, data: list[list[Any]]) -> None:
        lines = torch.tensor(data, dtype=torch.float32)
        self.x = lines[:, :-1]
        self.y = lines[:, -1]

    @classmethod
    def from_file(cls, file_name: str) -> TicTacToeDataset:
        lines = []
        with open(file_name, "r") as f:
            for line in f:
                lines.append([int(number) for number in line.strip().split()])

        return cls(lines)

    @classmethod
    def from_csv(cls, file_name: str) -> TicTacToeDataset:
        df = pd.read_csv(file_name, header=None)
        return cls(df.values.tolist())

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index].unsqueeze(0)

    def y_distribution(self) -> dict[float, int]:
        values, distribution = torch.unique(self.y, return_counts=True)
        return {v: d for v, d in zip(values.tolist(), distribution.tolist())}


def test_load_from_processing():
    """
    Testa criar um grafo de 5 nós, gerar <sample_amount> amostras para cores diferentes,
    por tudo na classe de dataset e dataloader
    """

    dataset = TicTacToeDataset.from_file("data/datasets/tictactoe/dados.trm")
    assert len(dataset) == 1000
    dataloader = DataLoader(dataset, 64, shuffle=True)
    input, output = next(iter(dataloader))
    assert list(input[0].size()) == [9]
    assert list(output[0].unsqueeze(0).size()) == [1, 1]
    print("Test 02 - OK")


if __name__ == "__main__":
    test_load_from_processing()
