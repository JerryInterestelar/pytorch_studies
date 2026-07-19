import torch
from torch import nn


# Como uma rede neural deve ser modelada usando o modelo base nn.Module()
# Essa é uma capaz de processar imagens 28x28 e retornar 10 possibilidades
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
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


device = (
    torch.accelerator.current_accelerator().type  # pyright: ignore
    if torch.accelerator.is_available()
    else "cpu"
)


# Faz todo o processo de forward na rede criada e faz a predição
def run_model():
    model = NeuralNetwork().to(device)
    X = torch.rand(1, 28, 28, device=device)

    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


# Pega os weights e biases para cada camada
def get_params():
    model = NeuralNetwork().to(device)
    for name, param in model.named_parameters():
        print(f"Camada: {name} | Tamanho: {param.size()} | Valores: {param[:2]}")


# Gera um tensor que simula o formado de 3 imagens 28x28
def images_samples() -> torch.Tensor:
    return torch.rand(3, 28, 28)


# Transforma cada matriz imagem MxN em um unico tensor de tamanho MxN
def flatten(samples: torch.Tensor) -> torch.Tensor:
    return nn.Flatten()(samples)


# Aplica uma transformação linear no tensor(achatado) usando os weights e os biases.
def linear(flatten_images: torch.Tensor) -> torch.Tensor:
    layer = nn.Linear(in_features=28 * 28, out_features=20)
    return layer(flatten_images)


# Aplica um transformação (rectified linear unit function) na camada calculada: max(0,x)
def relu(hidden_layer: torch.Tensor) -> torch.Tensor:
    return nn.ReLU()(hidden_layer)


# Corresponde a uma sequencia de módulos de transformações feitas em sequencia
# Retorna a pontuação bruta (logit) em um tensor
def sequential(input: torch.Tensor) -> torch.Tensor:
    return nn.Sequential(
        nn.Flatten(),  # Primeiro achata
        nn.Linear(
            in_features=28 * 28, out_features=20
        ),  # Transformação linear 28*28 -> 20
        nn.ReLU(),  # Transformação não linear
        nn.Linear(in_features=20, out_features=10),  # Transformação linear 20 -> 10
    )(input)  # tudo isso aplicado nesse tensor


# Transformação de normalização que converte o resultado entre -∞, +∞ -> 0, 1
def softmax(logit: torch.Tensor) -> torch.Tensor:
    return nn.Softmax()(logit)


if __name__ == "__main__":
    # imgs = images_samples()
    # flat_imgs = flatten(imgs)
    # hidden_layer = linear(flat_imgs)
    # relu_tranformed = relu(hidden_layer)
    # logit = sequential(imgs)
    # pred_probab = softmax(logit)

    # print(f"Imagens: {imgs}")
    # print(f"Imgens achatadas: {flat_imgs.size()}")
    # print(f"Transformação linear: {hidden_layer.size()}")
    # print(f"Antes Transformação ReLU: {hidden_layer}")
    # print(f"Antes Transformação ReLU: {relu_tranformed}")
    # print(f"Pontuação bruta: {logit}")
    # print(f"Probabilidade normalizada: {pred_probab}")
    get_params()
