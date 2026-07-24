import random
import pandas as pd

type Expressao = list[dict[int, bool]]


def gerar_variaveis(n_variaveis: int) -> dict[int, bool]:
    return {i: random.choice([True, False]) for i in range(n_variaveis)}


def tres_sat_expressao(n_variaveis: int, n_clausulas: int) -> Expressao:
    clausulas: Expressao = []

    for _ in range(n_clausulas):
        indice_trio = sorted(random.sample(range(n_variaveis), 3))
        clausulas.append({i: random.choice([True, False]) for i in indice_trio})

    return clausulas


def eval_tres_sat(variaveis: dict[int, bool], expressao: Expressao) -> bool:
    clausulas_analisadas = []
    for clausula in expressao:
        variaveis_analisadas = [
            variaveis[i] if value else not variaveis[i] for i, value in clausula.items()
        ]
        clausulas_analisadas.append(any(variaveis_analisadas))
    return all(clausulas_analisadas)


def print_expressao(expressao: Expressao) -> None:
    clausulas_formatadas = []

    for clausula in expressao:
        literais = [f"x{i}" if v else f"!x{i}" for i, v in clausula.items()]

        clausulas_formatadas.append(f"({' or '.join(literais)})")

    expressao_str = " and ".join(clausulas_formatadas)
    print(expressao_str)


def gerar_dataset(n_variaveis: int, n_clausulas: int, nome_arquivo: str):
    tres_sat_lista = tres_sat_expressao(n_variaveis, n_clausulas)
    print_expressao(tres_sat_lista)
    lines = []
    for _ in range(1000):
        x = gerar_variaveis(n_variaveis)
        line = [1 if value else 0 for value in x.values()]
        line.append(1 if eval_tres_sat(x, tres_sat_lista) else 0)
        lines.append(line)
    df = pd.DataFrame(lines)
    print(df)
    df.to_csv(nome_arquivo, index=False, header=True)


if __name__ == "__main__":
    gerar_dataset(8, 10, "tres_sat_dataset_8_10.csv")
