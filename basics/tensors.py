import torch
import numpy as np


def inicializing_tensors():
    data = [[1, 2], [3, 4]]
    np_array = np.array(data)
    x_data = torch.from_numpy(np_array)
    print(x_data)

    x_ones = torch.ones_like(x_data)
    print(x_ones)

    x_random = torch.rand_like(x_data, dtype=torch.float)
    print(x_random)


def operations():
    # Não funciona pra mim pq não tenho GPU dedicada aqui acho
    tensor = torch.ones(4, 4)
    if torch.accelerator.is_available():
        tensor = tensor.to(torch.accelerator.current_accelerator())
    else:
        print("Sem GPU :(")
    tensor[:, 1] = 0
    tensor[2, :] = 2
    tensor[1, 3] = 3

    print(f"{tensor}")
    print(f"Primeira linha: {tensor[0]}")
    print(f"Primeira coluna: {tensor[:, 0]}")
    print(f"ultima coluna: {tensor[..., -1]}")

    # Concatenar
    print(torch.cat([tensor, tensor, tensor], dim=1))


if __name__ == "__main__":
    operations()
