import random

type Color = tuple[int, int, int]


def get_random_color(n_colors: int) -> list[Color]:
    colors: list[Color] = []
    for _ in range(n_colors):
        r = random.randrange(0, 255)
        g = random.randrange(0, 255)
        b = random.randrange(0, 255)

        colors.append((r, g, b))

    return colors
