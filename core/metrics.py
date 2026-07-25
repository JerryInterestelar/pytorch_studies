import torch
from torch import Tensor


def binary_accuracy(predicted: Tensor, y: Tensor) -> float:
    predicted_classes = (predicted >= 0.0).type(torch.float)
    return (predicted_classes == y).type(torch.float).sum().item()


def multiclass_accuracy(predicted: Tensor, y: Tensor) -> float:
    return (predicted.argmax(dim=1) == y).type(torch.float).sum().item()
