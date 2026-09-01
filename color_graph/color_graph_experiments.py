import random

from torch.utils.data import DataLoader
from torch import nn
import torch

from color_graph.color_graph_network import ColorGraphNetwork
from color_graph.color_utils import get_colors
from color_graph.dataset_generator import ColorGraphDataset
from color_graph.graph_utils import Graph, GraphStructure
from color_graph.graph_generator import gen_diferent_graph_colors, squeese_dataset
from core.engine import train_loop, test_loop
from core.metrics import binary_accuracy


def make_random_graph_dataset(
    n_nodes: int, n_colors: int, edge_probability: float, sample_amount: int, slice: int
) -> tuple[Graph, ColorGraphDataset, ColorGraphDataset]:
    possible_colors = get_colors(n_colors)
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
    nodes: GraphStructure, n_colors: int, sample_amount: int, slice: int
) -> tuple[Graph, ColorGraphDataset, ColorGraphDataset]:
    n_nodes = len(nodes)
    possible_colors = get_colors(n_colors)
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


def load_model(
    n_nodes: int,
    n_colors: int,
) -> ColorGraphNetwork:
    print(
        f"* Carregando o modelo {n_nodes} nos X {n_colors} cores treinado com grafo X"
    )
    COLOR_GRAPH_MODEL_FILE = f"./data/models/color_graph/color_graph_model_{n_nodes}_nodes_{n_colors}_colors.pth"
    weights = torch.load(COLOR_GRAPH_MODEL_FILE, weights_only=True)

    model = ColorGraphNetwork(n_nodes * 3, 1)
    model.load_state_dict(weights)
    return model


def train_test_model(nodes: GraphStructure | None = None, n_colors=6, n_samples=1000):
    save_model = True
    learning_rate = 1e-3
    batch_size = 16
    epochs = 100
    n_nodes = 6
    slice = int((8 / 10) * n_samples)
    if nodes:
        n_nodes = len(nodes)
        graph, train_dataset, test_dataset = make_dataset(
            nodes, n_colors, n_samples, slice
        )
    else:
        graph, train_dataset, test_dataset = make_random_graph_dataset(
            n_nodes, n_colors, 0.2, n_samples, slice
        )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    color_graph_model = ColorGraphNetwork(n_nodes * 3, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(color_graph_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Iteração {t} -----------------")
        train_loop(
            train_dataloader,
            color_graph_model,
            loss_fn,
            optimizer,
            batch_size,
        )

    print("-" * 30)
    test_loop(test_dataloader, color_graph_model, loss_fn, binary_accuracy)
    if save_model:
        file_path = f"./data/models/color_graph/color_graph_model_{n_nodes}_nodes_{n_colors}_colors.pth"

        torch.save(color_graph_model.state_dict(), file_path)
        print(f"Modelo Salvo em {file_path}")

    print(graph.nodes)
    print(f"Distribuição treino: {train_dataset.y_distribution()}")
    print(f"Distribuição teste: {test_dataset.y_distribution()}")
    graph.show()
    return graph


def eval_color_graph_input(
    data_set: ColorGraphDataset,
    model: ColorGraphNetwork,
):
    model.eval()

    accuracy = 0
    with torch.no_grad():
        for x, y in data_set:
            prediction = model(x)
            accuracy += binary_accuracy(prediction, y)
    return accuracy / (len(data_set))


def single_example_analysis(nodes: GraphStructure, n_colors: int):
    n_nodes = len(nodes)
    possible_colors = get_colors(n_colors)
    assert possible_colors
    graph = Graph(nodes, possible_colors)
    n_samples = 100
    print(
        f"* Gerando um dataset simples de {n_samples} amostras para o calculo de acurácia"
    )
    graph_random_colors_dataset = ColorGraphDataset(
        squeese_dataset(gen_diferent_graph_colors(graph, possible_colors, n_samples))
    )
    graph_model = load_model(n_nodes, n_colors)
    print(f"* Mostrando a acurácia para o grafo: {graph.nodes}")
    accuracy = eval_color_graph_input(graph_random_colors_dataset, graph_model)
    print(
        f"A acurácia calculada para a avaliação das permutações de cores geradas pelo o dataset de {n_samples} amostras é de: {accuracy:.2f}%"
    )


if __name__ == "__main__":
    graphs = {
        "grafo_estrela": {0: [1, 2, 3, 4, 5], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0]},
        "grafo_anel": {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]},
        "grafo_arvore": {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]},
        "grafo_denso": {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3]},
    }
    n_colors = 3
    train_test_model(graphs["grafo_estrela"], n_colors, n_samples=1000)

    # single_example_analysis(graphs["grafo_anel"], n_colors)
