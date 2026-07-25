from typing import Sized, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tres_sat.dataset_generator import TreeSATDataset


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


def train_tree_sat_network(
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: nn.BCEWithLogitsLoss,
    optimizer: Optimizer,
    batch_size: int,
):
    dataset = cast(Sized, dataloader.dataset)
    size = len(dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # computar predição e perda
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 16 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Perda: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_tree_sat_network(
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: nn.BCEWithLogitsLoss,
):
    model.eval()

    dataset = cast(Sized, dataloader.dataset)
    size = len(dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_classes = (pred >= 0.0).type(torch.float)
            correct += (pred_classes == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Teste de erros: \n Acurácia: {(100 * correct):>0.1f}%, Perda média: {test_loss:>8f} \n"
    )


def main():

    learning_rate = 1e-3
    batch_size = 16
    epochs = 100
    dataset = TreeSATDataset("tres_sat_dataset_8_10.csv")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    tree_sat_model = TreeSATNetwork(8, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(tree_sat_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Iteração {t} -----------------")
        train_tree_sat_network(
            dataloader,
            tree_sat_model,
            loss_fn,
            optimizer,
            batch_size,
        )

    print("-" * 30)
    test_tree_sat_network(
        dataloader,
        tree_sat_model,
        loss_fn,
    )


if __name__ == "__main__":
    main()
