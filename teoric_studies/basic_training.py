import random
import math
import matplotlib.pyplot as plt


def dsigmoid(x: float) -> float:
    f = sigmoid(x)
    return f * (1 - f)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def treinamento(conjunto: list[tuple[int, int, int]]):
    delta = 0.5
    wa = random.random()
    wb = random.random()
    w1 = random.random()
    w2 = random.random()

    wa_samples = []
    wb_samples = []
    w1_samples = []
    w2_samples = []
    epoca = 0

    while epoca < 1000:
        dw1, dw2 = 0, 0
        random.shuffle(conjunto)
        for linha in conjunto:
            x1, x2, T = linha

            # Propagação
            n1 = x1 * w1 + x2 * w2
            n2 = n1 * wa + 1 * wb
            R = sigmoid(n2)

            # Retro propagação

            erro = -(T - R)
            dwa = erro * dsigmoid(n2) * n1
            dwb = erro * dsigmoid(n2)
            dw1 = erro * dsigmoid(n2) * wa * x1
            dw2 = erro * dsigmoid(n2) * wa * x2
            # Aprendizado
            wa -= delta * dwa
            wb -= delta * dwb
            w1 -= delta * dw1
            w2 -= delta * dw2

        epoca += 1
        # print(epoca)
        if epoca % 100 == 0:
            # print(w1, w2)
            wa_samples.append(wa)
            wb_samples.append(wb)
            w1_samples.append(w1)
            w2_samples.append(w2)
    return wa_samples, wb_samples, w1_samples, w2_samples


if __name__ == "__main__":
    conjunto = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    pesos = treinamento(conjunto)

    for x1, x2, _ in conjunto:
        resultado = x1 * pesos[-1][0] + x2 * pesos[-1][1]
        print(f"{x1}\t{x2}\t{x1 ^ x2}\t{resultado}")

    epocas = list(range(0, 1000, 100))
    _, ax = plt.subplots()
    ax.plot(epocas, pesos[0], label="wa")
    ax.plot(epocas, pesos[1], label="wb")
    ax.plot(epocas, pesos[2], label="w1")
    ax.plot(epocas, pesos[3], label="w2")
    ax.set(
        xlabel="epocas",
        ylabel="valor dos pesos",
        title="Valor dos pesos conforme as epocas",
    )
    ax.grid()
    plt.legend()
    plt.show()
