import torch
from torch import nn
from torch.utils.data import DataLoader

from core.engine import test_loop, train_loop
from core.metrics import binary_accuracy
from color_graph.dataset_generator import ColorGraphDataset


class ColorGraphNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.linear_relu_layers = nn.Sequential(
            # Camada de input -> Camada Escondida 1
            nn.Linear(n_inputs, n_inputs * 2),
            nn.ReLU(),
            # Camada Escondida 1 -> Camada Escondida 2
            nn.Linear(n_inputs * 2, n_inputs),
            nn.ReLU(),
            # Camada Escondida 2 -> Camada de output
            nn.Linear(n_inputs, n_outputs),
        )

    def forward(self, x):
        logits = self.linear_relu_layers(x)
        return logits


def main():

    save_model = False
    learning_rate = 1e-3
    batch_size = 16
    epochs = 100
    n_nodes = 5
    n_colors = 5
    train_dataset = ColorGraphDataset.from_csv(
        f"./data/datasets/color_graph/train_{n_nodes}_nodes_{n_colors}_colors.csv",
    )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = ColorGraphDataset.from_csv(
        f"./data/datasets/color_graph/test_{n_nodes}_nodes_{n_colors}_colors.csv",
    )
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    tree_sat_model = ColorGraphNetwork(n_nodes * 3, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(tree_sat_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Iteração {t} -----------------")
        train_loop(
            train_dataloader,
            tree_sat_model,
            loss_fn,
            optimizer,
            batch_size,
        )

    print("-" * 30)
    test_loop(test_dataloader, tree_sat_model, loss_fn, binary_accuracy)
    if save_model:
        file_path = f"./data/models/color_graph/color_graph_model_{n_nodes}_nodes_{n_colors}_colors.pth"

        torch.save(tree_sat_model.state_dict(), file_path)
        print(f"Modelo Salvo em {file_path}")


if __name__ == "__main__":
    main()
