from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from color_graph.color_utils import get_colors
from color_graph.graph_generator import gen_diferent_graph_colors, squeese_dataset
from color_graph.graph_utils import Graph


class ColorGraphDataset(Dataset):
    def __init__(self, data: list[list[Any]]) -> None:
        lines = torch.tensor(data, dtype=torch.float32)
        self.x = lines[:, :-1]
        self.y = lines[:, -1]

    @classmethod
    def from_csv(cls, file_name: str) -> ColorGraphDataset:
        df = pd.read_csv(file_name, header=None)
        return cls(df.values.tolist())

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index].unsqueeze(0)

    def y_distribution(self):
        return torch.unique(self.y, return_counts=True)


def test_load_csv():
    """
    Testa criar ler o dataset já criado em "graph_generator" e se o csv gerado encaixa na classe de dataset e dataloader
    Pra esse teste ficar bom mesmo eu precisaria criar o csv aqui antes e ler ele
    """
    dataset = ColorGraphDataset.from_csv("./data/datasets/color_graph/test.csv")
    assert len(dataset) == 200
    dataloader = DataLoader(dataset, 64, shuffle=True)
    input, output = next(iter(dataloader))
    assert list(input[0].size()) == [15]
    assert list(output[0].unsqueeze(0).size()) == [1, 1]
    print("Test 01 - OK")


def test_load_from_processing():
    """
    Testa criar um grafo de 5 nós, gerar <sample_amount> amostras para cores diferentes,
    por tudo na classe de dataset e dataloader
    """
    n = 5
    possible_colors = get_colors(4)
    graph = Graph.random(n, 0.5, possible_colors)
    sample_amount = 10
    raw_data = squeese_dataset(
        gen_diferent_graph_colors(graph, possible_colors, sample_amount)
    )
    dataset = ColorGraphDataset(raw_data)
    assert len(dataset) == sample_amount
    dataloader = DataLoader(dataset, 64, shuffle=True)
    input, output = next(iter(dataloader))
    assert list(input[0].size()) == [15]
    assert list(output[0].unsqueeze(0).size()) == [1, 1]
    print("Test 02 - OK")


if __name__ == "__main__":
    test_load_csv()
    test_load_from_processing()
