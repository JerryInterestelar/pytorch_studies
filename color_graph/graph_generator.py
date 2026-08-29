import random
import pandas as pd


from color_graph.color_utils import get_colors
from color_graph.graph_utils import Graph, GraphStructure


def basic_graph(graph_dict: GraphStructure | None = None, colors: list | None = None):
    if graph_dict and colors:
        graph = Graph(graph_dict, colors)
    else:
        graph = Graph.random(5)
    print(graph)

    assert graph

    def compare_nodes(node, neighbor):
        result = graph.colors[node] == graph.colors[neighbor]
        print(f"Nó {node} -> Vizinho {neighbor}: {'igual' if result else 'diferente'}")
        return result

    result = graph.map_all_edges(compare_nodes)
    print("MESTRE" if not any(result) else "Grafo comum")
    graph.show()


def gen_diferent_graph_colors(
    graph: Graph | None, possible_colors: list | None, len: int
) -> list[list]:
    assert graph
    assert possible_colors
    rows = []
    for _ in range(len):
        new_color = random.choices(possible_colors, k=n)
        graph.set_colors(new_color)
        result = graph.map_all_edges(lambda n, e: graph.colors[n] == graph.colors[e])
        rows.append([graph.colors, 1.0 if not any(result) else 0.0])
    return rows


def save_to_csv(
    rows: list, slice_point: int, train_file_name: str, test_file_name: str
):
    data = []
    for colors, value in rows:
        line = []
        for r, g, b in colors:
            line.extend([r, g, b])
        line.append(value)
        data.append(line)

    df = pd.DataFrame(data)
    print(f"Incidência de valores 1 e 0: {df.iloc[:, -1].value_counts()}")
    df[:slice_point].to_csv(train_file_name, index=False, header=False)
    df[slice_point:].to_csv(test_file_name, index=False, header=False)
    print(f"CSVs salvos em {train_file_name} e {test_file_name}")


if __name__ == "__main__":
    n = 5
    possible_colors = get_colors(4)
    graph = Graph.random(n, possible_colors)
    save_to_csv(
        gen_diferent_graph_colors(graph, possible_colors, 1000),
        800,
        "./data/datasets/color_graph/train.csv",
        "./data/datasets/color_graph/test.csv",
    )
