import torch
from torch import nn
from torch.utils.data import DataLoader

from core.engine import test_loop, train_loop
from core.metrics import binary_accuracy
from color_graph.dataset_generator import ColorGraphDataset


class ColorGraphNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        layers = [
            # Camada de input -> Camada Escondida 1
            nn.Linear(n_inputs, n_inputs * 2),
            nn.ReLU(),
            # Camada Escondida 1 -> Camada Escondida 2
            nn.Linear(n_inputs * 2, n_inputs),
            nn.ReLU(),
            # Camada Escondida 2 -> Camada de output
            nn.Linear(n_inputs, n_outputs),
        ]
        self.linear_relu_layers = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_layers(x)
        return logits


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


def save_model(n_nodes: int, n_colors: int, model: ColorGraphNetwork):
    file_path = f"./data/models/color_graph/color_graph_model_{n_nodes}_nodes_{n_colors}_colors.pth"

    torch.save(model.state_dict(), file_path)
    print(f"Modelo Salvo em {file_path}")


def train_test_color_graph_model(
    datasets: tuple[ColorGraphDataset, ColorGraphDataset], n_nodes: int
) -> ColorGraphNetwork:
    learning_rate = 1e-3
    batch_size = 16
    epochs = 100
    train_dataset, test_dataset = datasets

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
    return color_graph_model


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


if __name__ == "__main__":
    ...
