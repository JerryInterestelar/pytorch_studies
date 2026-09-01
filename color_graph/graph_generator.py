import pandas as pd

from color_graph.graph_utils import Graph, GraphStructure, get_color_set


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


def gen_color_dataset(graph: Graph, size: int) -> list[list]:
    rows = []
    for _ in range(size):
        new_color = get_color_set(len(graph.nodes))
        graph.set_colors(new_color)

        result = graph.map_all_edges(lambda n, e: graph.colors[n] == graph.colors[e])

        # WARN: Se a quantidade de cores vier a ser diferente um dia, mudar isso
        coded_colors = []
        for color in graph.colors:
            match color:
                case "red":
                    coded_colors.append(-1.0)
                case "green":
                    coded_colors.append(0.0)
                case "blue":
                    coded_colors.append(1.0)

        rows.append([coded_colors, 1.0 if not any(result) else 0.0])
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
    print(gen_color_dataset(Graph.random(5, 0.3), 10))
