import random

type Color = tuple[float, float, float]


def get_colors(n_colors: int) -> list[Color] | None:
    colors = [
        # Primárias
        (1.0, 0.0, 0.0),  # Vermelho
        (0.0, 1.0, 0.0),  # Verde
        (0.0, 0.0, 1.0),  # Azul
        # Secundárias
        (1.0, 1.0, 0.0),  # Amarelo
        (0.0, 1.0, 1.0),  # Ciano
        (1.0, 0.0, 1.0),  # Magenta
        # Compostas
        (1.0, 0.5, 0.0),  # Laranja
        (0.5, 0.0, 0.5),  # Roxo
        (1.0, 0.75, 0.8),  # Rosa
        (0.6, 0.3, 0.0),  # Marrom
        # Neutras
        (0.5, 0.5, 0.5),  # Cinza
        (0.0, 0.0, 0.0),  # Preto
        (1.0, 1.0, 1.0),  # Branco
    ]
    if n_colors > len(colors):
        print(
            f"Número de cores especificados '{n_colors}' superior ao número de cores disponível '{len(colors)}', tente usar cores aleatórias com a função 'get_random_color' talvez para grafos maiores"
        )
        return
    return colors[:n_colors]


def get_random_color(n_colors: int) -> list[Color]:
    colors: list[Color] = []
    for _ in range(n_colors):
        while True:
            r = round(random.random(), 2)
            g = round(random.random(), 2)
            b = round(random.random(), 2)
            color = (r, g, b)
            if color not in colors:
                colors.append(color)
                break

    return colors
