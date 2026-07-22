import random


def tres_sat(n_variaveis: int, n_clausulas: int) -> list[dict[int, bool]]:
    x: dict[int, bool] = {}
    clausulas: list[dict] = []
    for i in range(n_variaveis):
        x[i] = random.choice([True, False])

    for i in range(n_clausulas):
        trio_keys = sorted(random.sample(tuple(x.keys()), 3))
        trio = [x[j] for j in trio_keys]

        clausulas.append(dict(zip(trio_keys, trio)))

    return clausulas


if __name__ == "__main__":
    tres_sat_lista = tres_sat(10, 10)
    expressao = ") and (".join(
        (
            " or ".join(f"x{i}" if v else f"!x{i}" for i, v in clausula.items())
            for clausula in tres_sat_lista
        )
    )

    print(f"({expressao})")
