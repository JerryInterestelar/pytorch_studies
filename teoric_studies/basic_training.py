import random
import math
import matplotlib.pyplot as plt


def dsigmoid(x: float) -> float:
    return x * (1.0 - x)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def treinamento(conjunto: list[tuple[int, int, int]]):
    delta = 0.01
    n_pesos = 6
    w = [random.uniform(-0.5, 0.5) for _ in range(n_pesos)]
    d = [0.0 for _ in range(n_pesos)]

    w_samples = [list() for _ in range(n_pesos)]

    epoca = 0

    while epoca < 1000:
        for linha in conjunto:
            x1, x2, T = linha

            # Propagação
            s1 = x1 * w[0] + x2 * w[2]
            s2 = x1 * w[1] + x2 * w[3]
            s3 = s1 * w[4] + s2 * w[5]

            # retro Propagação
            erro = -(T - s3)
            d[5] = erro * s2
            d[4] = erro * s1

            d[3] = erro * w[5] * x2
            d[2] = erro * w[4] * x2
            d[1] = erro * w[5] * x1
            d[0] = erro * w[4] * x1

            for i in range(len(w)):
                w[i] -= delta * d[i]
        if epoca % 100 == 0:
            for i in range(len(w)):
                w_samples[i].append(w[i])
        epoca += 1
    return w_samples


if __name__ == "__main__":
    conjunto = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    pesos = treinamento(conjunto)

    pesos_finais = [peso[-1] for peso in pesos]

    for x1, x2, y in conjunto:
        s1 = x1 * pesos_finais[0] + x2 * pesos_finais[2]
        s2 = x1 * pesos_finais[1] + x2 * pesos_finais[3]
        s3 = s1 * pesos_finais[4] + s2 * pesos_finais[5]
        print(f"{x1}\t{x2}\t{y}\t{s3}")

    epocas = list(range(0, 1000, 100))
    _, ax = plt.subplots()
    for i, amostra in enumerate(pesos, 1):
        ax.plot(epocas, amostra, label=f"w{i}")
    ax.set(
        xlabel="epocas",
        ylabel="valor dos pesos",
        title="Valor dos pesos conforme as epocas - XNOR Modificado",
    )
    ax.grid()
    plt.legend()
    plt.show()
