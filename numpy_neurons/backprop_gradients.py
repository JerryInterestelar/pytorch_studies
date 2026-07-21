import numpy as np

"""
Ilustra o processo de Forward e Backpropagation para a geração dos vetores gradientes para o ajustes de pesos e vieses
Implentação em python da matemáticas descrita do video https://www.youtube.com/watch?v=tIeHLnjs5U8&t=16s
"""


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z):
    return sigmoid(z) * (1 - sigmoid(z))


def main():
    n_in = 3
    n_hidden = 2
    n_out = 3

    input_a = np.array([[1], [2], [3]])
    desired_output = np.array([[1], [2], [3]])

    # Forward

    w_l1 = np.random.randn(n_hidden, n_in)
    b_l1 = np.zeros((n_hidden, 1))
    z_l1 = w_l1 @ input_a + b_l1
    a_l1 = sigmoid(z_l1)

    w_l = np.random.rand(n_out, n_hidden)
    b_l = np.zeros((n_out, 1))
    z_l = w_l @ a_l1 + b_l
    a_l = sigmoid(z_l)

    # Custo da rede
    cost_0 = (desired_output - a_l) ** 2
    print(f"Custo total: {cost_0}")

    # Backpropagation
    # Chain rule
    d_cost_0 = (a_l - desired_output) * 2
    d_a_l = sigmoid_derivada(z_l)

    delta_l = d_cost_0 * d_a_l

    # gradiente - camada de saida
    d_c0_d_w_l = delta_l @ a_l1.T
    d_c0_d_b_l = delta_l

    delta_l1 = (w_l.T @ delta_l) * sigmoid_derivada(z_l1)

    d_c0_d_w_l1 = delta_l1 @ input_a.T
    d_c0_d_b_l1 = delta_l1

    gradients_w = [d_c0_d_w_l1, d_c0_d_w_l]
    gradients_b = [d_c0_d_b_l1, d_c0_d_b_l]
    print(f"Gradientes dos pesos: {gradients_w}")
    print(f"Gradientes dos vieses: {gradients_b}")


if __name__ == "__main__":
    main()
