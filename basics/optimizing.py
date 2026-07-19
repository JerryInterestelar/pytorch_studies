import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from typing import Sized, Tuple, cast

learning_rate = 1e-3
batch_size = 64
epochs = 10

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: Optimizer,
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Perda: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: nn.CrossEntropyLoss,
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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Teste de erros: \n Acurácia: {(100 * correct):>0.1f}%, Perda média: {test_loss:>8f} \n"
    )


def main():

    model = NeuralNetwork()

    # Trainning/Optimization loop
    # Faz o calculo da diferença entre o valor esperado e o logit com Negative Log Likelihood + LogSoftmax
    loss_fn = nn.CrossEntropyLoss()
    # Cria um objeto de otimização baseado nos parametros da rede e o learning_rate com Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Época {t + 1} \n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    # Saving the model
    torch.save(model.state_dict(), "basic_model_weights.pth")
    print("Pronto")


if __name__ == "__main__":
    main()
