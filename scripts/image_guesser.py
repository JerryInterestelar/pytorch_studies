import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision import datasets
from torch import Tensor, nn

from basics.neural_network import NeuralNetwork


fashion_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
weights = torch.load("basic_model_weights.pth", weights_only=True)

model = NeuralNetwork()
model.load_state_dict(weights)


def show_image(dataset: datasets.MNIST, index: int = 0):
    image, label = dataset[index]

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Imagem real: {dataset.classes[label]}")
    plt.show()


def predict_image(
    dataset: datasets.MNIST,
    model: NeuralNetwork,
    index: int,
) -> tuple[int, float]:
    model.eval()

    image: Tensor
    image, _ = dataset[index]
    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        predicion: Tensor = model(image_batch)
        probability = nn.Softmax(dim=1)(predicion)
        predicted_class = int(probability.argmax(1).item())
    return predicted_class, probability.squeeze()[predicted_class]


if __name__ == "__main__":
    image_index = 42
    predict_class, probability = predict_image(fashion_dataset, model, image_index)
    print(
        "A imagem {} selecionada é: {} com {:>0.1f} % de certeza".format(
            image_index, fashion_dataset.classes[predict_class], probability * 100
        )
    )
    show_image(fashion_dataset, image_index)
