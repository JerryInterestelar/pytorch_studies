import numpy as np

"""
Gerado por IA - Ilustra como o processo visto em backprop_gradients pode ser generalizado para várias camadas
"""


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z):
    return sigmoid(z) * (1 - sigmoid(z))


tamanhos_camadas = [3, 4, 4, 3]

pesos = [
    np.random.randn(y, x) for x, y in zip(tamanhos_camadas[:-1], tamanhos_camadas[1:])
]
vieses = [np.zeros((y, 1)) for y in tamanhos_camadas[1:]]

input_x = np.array([[1], [2], [3]])
desired_y = np.array([[1], [2], [3]])

ativacoes = [input_x]  # Guardamos o input como a "ativação 0"
zs = []  # Guardamos todos os Zs para usar na derivada depois

ativacao_atual = input_x

for w, b in zip(pesos, vieses):
    z = w @ ativacao_atual + b
    zs.append(z)

    ativacao_atual = sigmoid(z)
    ativacoes.append(ativacao_atual)


gradientes_w = [np.zeros(w.shape) for w in pesos]
gradientes_b = [np.zeros(b.shape) for b in vieses]

delta = 2 * (ativacoes[-1] - desired_y) * sigmoid_derivada(zs[-1])

gradientes_w[-1] = delta @ ativacoes[-2].T
gradientes_b[-1] = delta

for L in range(2, len(tamanhos_camadas)):
    z = zs[-L]
    derivada = sigmoid_derivada(z)

    delta = (pesos[-L + 1].T @ delta) * derivada

    gradientes_w[-L] = delta @ ativacoes[-L - 1].T
    gradientes_b[-L] = delta
