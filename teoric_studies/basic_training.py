import random
import matplotlib.pyplot as plt


def treinamento(conjunto: list[tuple[int, int, int]]):
    delta = 1e-2
    w_1 = random.random()
    w_2 = random.random()

    w_1_samples = []
    w_2_samples = []
    epoca = 0

    while epoca < 1000:
        d_w_1, d_w_2 = 0, 0
        random.shuffle(conjunto)
        for linha in conjunto:
            x_1, x_2, T = linha

            # Propagação
            R = x_1 * w_1 + x_2 * w_2

            # Retro propagação

            d_w_1 = -(T - R) * x_1
            d_w_2 = -(T - R) * x_2
        # Aprendizado
        w_1 -= delta * d_w_1
        w_2 -= delta * d_w_2
        epoca += 1
        # print(epoca)
        if epoca % 100 == 0:
            # print(w_1, w_2)
            w_1_samples.append(w_1)
            w_2_samples.append(w_2)
    return w_1_samples, w_2_samples


if __name__ == "__main__":
    conjunto = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    pesos = treinamento(conjunto)
    epocas = list(range(0, 1000, 100))
    _, ax = plt.subplots()
    ax.plot(epocas, pesos[0], label="w_1")
    ax.plot(epocas, pesos[1], label="w_2")
    ax.set(
        xlabel="epocas",
        ylabel="valor dos pesos",
        title="Valor dos pesos conforme as epocas",
    )
    ax.grid()
    plt.legend()
    plt.show()
