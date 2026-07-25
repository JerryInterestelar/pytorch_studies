from typing import Callable, Sized, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

type Loss = nn.modules.loss._Loss


def train_loop(
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: Loss,
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
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % int(size / 100) == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Perda: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: Loss,
    metric_fn: Callable[[Tensor, Tensor], float],
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
            correct += metric_fn(pred, y)

    test_loss /= num_batches
    correct /= size
    print(
        f"Teste de erros: \n Acurácia: {(100 * correct):>0.1f}%, Perda média: {test_loss:>8f} \n"
    )
