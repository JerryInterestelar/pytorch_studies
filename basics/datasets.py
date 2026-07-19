from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch


# Como funciona uma classe de dataset modificado
# Futuramente ao desenvolver datasets mais especificos deve ser interessante implementar eles seguindo
# essa lógica para ter o máximo de compatibilidade com a API do torch
#
class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[index, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def test():

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    train_dataloader = DataLoader(training_data, 64, True)
    test_dataloader = DataLoader(test_data, 64, True)

    train_features: torch.Tensor
    test_features: torch.Tensor

    train_features, train_labels = next(iter(train_dataloader))
    test_features, _ = next(iter(test_dataloader))

    print(f"Train features shape: {train_features.size()}")
    print(f"Test features shape: {test_features.size()}")

    # Pegando uma amostra aleatória
    img = train_features[0].squeeze()
    label = train_labels[0].item()

    plt.imshow(
        img,
        cmap="gray",
    )
    plt.title(label)
    plt.show()


if __name__ == "__main__":
    test()
