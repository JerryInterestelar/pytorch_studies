import torch
from torch import nn
from torch.utils.data import DataLoader

from core.engine import test_loop, train_loop
from core.metrics import binary_accuracy
from tree_sat.dataset_generator import TreeSATDataset


class TreeSATNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.linear_relu_layers = nn.Sequential(
            # Camada de input -> Camada Escondida 1
            nn.Linear(n_inputs, 16),
            nn.ReLU(),
            # Camada Escondida 1 -> Camada Escondida 2
            nn.Linear(16, 8),
            nn.ReLU(),
            # Camada Escondida 2 -> Camada de output
            nn.Linear(8, n_outputs),
        )

    def forward(self, x):
        logits = self.linear_relu_layers(x)
        return logits


def main():

    save_model = True
    learning_rate = 1e-3
    batch_size = 16
    epochs = 100
    input_n = 8
    dataset = TreeSATDataset("./data/datasets/tres_sat_dataset_8_10.csv")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    tree_sat_model = TreeSATNetwork(input_n, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(tree_sat_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Iteração {t} -----------------")
        train_loop(
            dataloader,
            tree_sat_model,
            loss_fn,
            optimizer,
            batch_size,
        )

    print("-" * 30)
    test_loop(dataloader, tree_sat_model, loss_fn, binary_accuracy)
    if save_model:
        file_path = f"./data/models/tree_sat_model_{input_n}x1_weights.pth"

        torch.save(tree_sat_model.state_dict(), file_path)
        print(f"Modelo Salvo em {file_path}")


if __name__ == "__main__":
    main()
