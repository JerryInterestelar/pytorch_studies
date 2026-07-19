import torch
from torch import nn
from basics.optimizing import NeuralNetwork, test_loop, test_dataloader


# Carregamos o modelo que treinamos e salvamos na função optimizing
weights = torch.load("basic_model_weights.pth", weights_only=True)

model = NeuralNetwork()
model.load_state_dict(weights)
loss_fn = nn.CrossEntropyLoss()

test_loop(test_dataloader, model, loss_fn)
