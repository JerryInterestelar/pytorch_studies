import torch


def test():

    print("=== Status da Instalação ===")
    print(f"Versão do PyTorch: {torch.__version__}")
    print(
        f"CUDA disponível? {torch.cuda.is_available()} (Esperado: False para seu setup)\n"
    )

    print("=== Teste de Tensores ===")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0])

    print(f"Tensor x: {x}")
    print(f"Tensor y: {y}\n")

    print("=== Teste do Autograd ===")
    z = (x * y).sum()
    print(f"Resultado de (x * y).sum(): {z.item()}")

    z.backward()

    print(f"Gradiente de x (dz/dx): {x.grad}")

    assert x.grad is not None

    if torch.equal(x.grad, y):
        print("\nSucesso: O Autograd calculou a derivada corretamente.")
    else:
        print("\nErro: O gradiente calculado está incorreto.")


def test02():
    x = torch.rand(5, 3)
    print(x)


if __name__ == "__main__":
    test02()
