import random
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from color_graph.color_utils import get_colors
from color_graph.graph_generator import gen_diferent_graph_colors, squeese_dataset
from color_graph.graph_utils import Graph, GraphStructure


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


def make_random_graph_dataset(
    n_nodes: int,
    n_possible_colors: int,
    edge_probability: float,
    sample_amount: int,
    slice: int,
) -> tuple[Graph, ColorGraphDataset, ColorGraphDataset]:
    possible_colors = get_colors(n_possible_colors)
    graph = Graph.random(n_nodes, edge_probability, possible_colors)
    assert graph
    raw_data = squeese_dataset(
        gen_diferent_graph_colors(graph, possible_colors, sample_amount)
    )
    return (
        graph,
        ColorGraphDataset(raw_data[:slice]),
        ColorGraphDataset(raw_data[slice:]),
    )


def make_dataset(
    nodes: GraphStructure, n_possible_colors: int, sample_amount: int, slice: int
) -> tuple[Graph, ColorGraphDataset, ColorGraphDataset]:
    n_nodes = len(nodes)
    possible_colors = get_colors(n_possible_colors)
    assert possible_colors
    graph = Graph(nodes, random.choices(possible_colors, k=n_nodes))
    assert graph
    raw_data = squeese_dataset(
        gen_diferent_graph_colors(graph, possible_colors, sample_amount)
    )
    return (
        graph,
        ColorGraphDataset(raw_data[:slice]),
        ColorGraphDataset(raw_data[slice:]),
    )


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
    _, dataset, _ = make_random_graph_dataset(
        n_nodes=5, n_possible_colors=4, edge_probability=0.5, sample_amount=10, slice=8
    )
    assert len(dataset) == 8
    dataloader = DataLoader(dataset, 64, shuffle=True)
    input, output = next(iter(dataloader))
    assert list(input[0].size()) == [15]
    assert list(output[0].unsqueeze(0).size()) == [1, 1]
    print("Test 02 - OK")


if __name__ == "__main__":
    test_load_csv()  # Falha se não tiver o arquivo
    test_load_from_processing()
