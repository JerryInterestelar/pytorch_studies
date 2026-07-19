import torch


def calc_loss(input: torch.Tensor, expected_output: torch.Tensor) -> None:
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    # Forward basico de 1 neuronio
    # logit
    z = torch.matmul(input, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, expected_output)

    loss.backward()
    print(w.grad)
    print(b.grad)


if __name__ == "__main__":
    # Input
    x = torch.ones(5)
    # Expected output
    y = torch.zeros(3)

    calc_loss(x, y)
