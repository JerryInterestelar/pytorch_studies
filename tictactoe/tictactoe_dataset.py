from typing import Any


def tictactoe_dataset() -> list[list[Any]]:
    lines = []
    with open("data/datasets/tictactoe/dados.trm", "r") as f:
        for line in f:
            lines.append([int(number) for number in line.strip().split()])

    return lines


if __name__ == "__main__":
    print(tictactoe_dataset())
