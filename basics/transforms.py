import torch.nn.functional as F
import torch
from torchvision.transforms import v2
from torchvision import datasets


def test():

    _ = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        # ToImage converte de PIL ou ndarray -> torchvision.tv_tensors.Image
        # ToDtype converte pra float32 com a intesidade do pixel para [0, 1]
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        target_transform=v2.Lambda(
            lambda y: F.one_hot(torch.Tensor(y), num_classes=10).float()
        ),
    )


if __name__ == "__main__":
    test()
