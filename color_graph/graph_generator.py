import random
import matplotlib.pyplot as plt
import networkx as nx
from graph_utils import map_graph_edges, get_graph_dict, random_graph_edge_list  # noqa: F401


def show_graph(graph: dict, colors: list):
    G = nx.DiGraph(graph)
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        with_labels=True,
        node_color=colors,
        node_size=800,
        font_color="white",
        font_weight="bold",
        arrowsize=20,
    )
    plt.show()


def main():
    # graph = {0: [1], 1: [2, 3], 2: [3, 4], 3: [4, 0], 4: [0, 1]}
    n = 20
    graph = get_graph_dict(random_graph_edge_list(n))
    colors = random.choices(["red", "green", "blue", "pink", "purple"], k=n)

    def compare_edges(node, neighbor):
        print(f"Nó: {node} -> {neighbor}")
        return colors[node] == colors[neighbor]

    result = map_graph_edges(graph, 0, compare_edges)
    print(result)
    print("Grafo mestre" if not any(result) else "Grafo Comum")

    show_graph(graph, colors)


if __name__ == "__main__":
    main()
