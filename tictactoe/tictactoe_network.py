import torch
from torch import nn
from torch.utils.data import DataLoader

from tictactoe.dataset_generator import TicTacToeDataset
from core.engine import train_loop, test_loop

from core.metrics import single_output_accuracy


class TicTacToeNetwork(nn.Module):
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


def train_test_tictactoe_model(
    datasets: tuple[TicTacToeDataset, TicTacToeDataset],
) -> TicTacToeNetwork:
    learning_rate = 1e-3
    batch_size = 16
    epochs = 100
    train_dataset, test_dataset = datasets

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    tictactoe_model = TicTacToeNetwork(len(train_dataset[0][0]), 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(tictactoe_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Iteração {t} -----------------")
        train_loop(
            train_dataloader,
            tictactoe_model,
            loss_fn,
            optimizer,
            batch_size,
        )

    print("-" * 30)
    test_loop(test_dataloader, tictactoe_model, loss_fn, single_output_accuracy)
    return tictactoe_model


def eval_tictactoe_input(
    data_set: TicTacToeDataset,
    model: TicTacToeNetwork,
):
    model.eval()

    with torch.no_grad():
        for x, y in data_set:
            print(model(x), y)


if __name__ == "__main__":
    train_dataset = TicTacToeDataset.from_file("data/datasets/tictactoe/dados.trm")
    test_dataset = TicTacToeDataset.from_file("data/datasets/tictactoe/dados.tst")
    model = train_test_tictactoe_model((train_dataset, test_dataset))
    # eval_tictactoe_input(test_dataset, model)
