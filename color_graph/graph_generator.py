import random
from itertools import product

import pandas as pd

from color_graph.graph_utils import (
    Graph,
    GraphStructure,
    get_color_set,
    POSSIBLE_COLORS,
)


def basic_graph(graph_dict: GraphStructure | None = None) -> Graph:
    if graph_dict:
        graph = Graph(graph_dict)
    else:
        graph = Graph.random(5, 0.4)
    print(graph)

    def compare_nodes(node, neighbor):
        result = graph.colors[node] == graph.colors[neighbor]
        print(f"Nó {node} -> Vizinho {neighbor}: {'igual' if result else 'diferente'}")
        return result

    result = graph.map_all_edges(compare_nodes)
    print("MESTRE" if not any(result) else "Grafo comum")
    graph.show()
    return graph


def is_valid(graph: Graph) -> float:
    result = graph.map_all_edges(lambda n, e: graph.colors[n] == graph.colors[e])
    return 1.0 if not any(result) else 0.0


def gen_balanced_color_dataset(graph: Graph, size: int) -> tuple[list[list], tuple]:
    total_combinations = 3 ** len(graph.nodes)

    COMB_LIMIT = 5_000_000

    if total_combinations > COMB_LIMIT:
        raise ValueError(
            f"Muitas combinações. Um grafo de {len(graph.nodes)} nós e 3 cores "
            f"geraria {total_combinations} combinações na memória, o que pode travar o computador."
        )
    color_combinations = product(
        list(POSSIBLE_COLORS.values()), repeat=len(graph.nodes)
    )

    valid_colors = []
    invalid_colors = []

    for combination in color_combinations:
        graph.set_colors(list(combination))
        result = is_valid(graph)
        if result:
            valid_colors.append([list(combination), result])
        else:
            invalid_colors.append([list(combination), result])

    valid_len = len(valid_colors)
    invalid_len = len(invalid_colors)
    if valid_len == 0:
        raise ValueError("Não foi possivel gerar cores válidas")
    if invalid_len == 0:
        raise ValueError("Não foi possivel gerar cores inválidas")

    valid_dataset = random.choices(valid_colors, k=round(size / 2))
    invalid_dataset = random.choices(invalid_colors, k=round(size / 2))
    dataset = valid_dataset + invalid_dataset
    random.shuffle(dataset)
    return dataset, (valid_len, invalid_len)


def gen_color_dataset(graph: Graph, size: int) -> list[list]:
    rows = []
    for _ in range(size):
        new_color = get_color_set(len(graph.nodes))
        graph.set_colors(new_color)

        coded_colors = [POSSIBLE_COLORS[color] for color in graph.colors]
        rows.append([coded_colors, is_valid(graph)])
    return rows


def squeese_dataset(rows: list[list]) -> list[list]:
    data = []
    for colors, value in rows:
        line = []
        for c in colors:
            line.extend([c])
        line.append(value)
        data.append(line)
    return data


def save_to_csv(
    rows: list, slice_point: int, train_file_name: str, test_file_name: str
):
    data = squeese_dataset(rows)
    df = pd.DataFrame(data)
    print(f"Incidência de valores 1 e 0: {df.iloc[:, -1].value_counts()}")
    df[:slice_point].to_csv(train_file_name, index=False, header=False)
    df[slice_point:].to_csv(test_file_name, index=False, header=False)
    print(f"CSVs salvos em {train_file_name} e {test_file_name}")


def make_dataset():
    n_nodes = 5
    graph = Graph.random(n_nodes, 0.2)
    save_to_csv(
        gen_color_dataset(graph, 1000),
        800,
        f"./data/datasets/color_graph/train_{n_nodes}_nodes.csv",
        f"./data/datasets/color_graph/test_{n_nodes}_nodes.csv",
    )


if __name__ == "__main__":
    graph = Graph.random(5, 0.4)
    # print(gen_color_dataset(graph, 10))
    gen_balanced_color_dataset(graph, 100)

    # graph.show()
