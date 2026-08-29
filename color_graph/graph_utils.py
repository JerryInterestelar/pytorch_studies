import random
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from color_graph.color_utils import get_colors

GraphStructure = dict[Any, list[Any]]


class Graph:
    def __init__(self, nodes: GraphStructure, colors: list) -> None:
        self._nodes: GraphStructure = nodes
        self._colors = colors

    @classmethod
    def random(cls, n: int, possible_colors: list | None = None) -> Graph | None:
        if possible_colors:
            chosen_colors = random.choices(possible_colors, k=n)
        else:
            if not (possible_colors := get_colors(3)):
                print(
                    "Muitas cores pedidas, use 'get_random_colors' ou adicione mais cores manualmente"
                )
                return

            chosen_colors = random.choices(possible_colors, k=n)
        matrix = random_graph_matrix(n)
        graph = cls(get_graph_dict(matrix), chosen_colors)
        return graph

    @property
    def nodes(self) -> GraphStructure:
        return self._nodes

    @property
    def colors(self) -> list:
        return self._colors

    @property
    def as_ndarray_matrix(self) -> np.ndarray:
        return np.array(get_graph_matrix(self._nodes))

    @property
    def as_ndarray_edge_list(self) -> np.ndarray:
        return get_graph_ndarray(self._nodes)

    def set_colors(self, colors: list):
        self._colors = colors

    def map_dfs_edges(self, start: Any, function: Callable[[Any, Any], Any]) -> list:
        return map_dfs_edges(self._nodes, start, function)

    def map_all_edges(self, function: Callable[[Any, Any], Any]) -> list:
        return map_all_edges(self._nodes, function)

    def find_path(self, start: Any, end: Any) -> list | None:
        path = []
        return find_path(self._nodes, start, end, path)

    def find_all_paths(self, start: Any, end: Any) -> list | None:
        path = []
        return find_all_paths(self._nodes, start, end, path)

    def find_shortest_path(self, start: Any, end: Any):
        path = []
        return find_shortest_path(self._nodes, start, end, path)

    def show(self):
        G = nx.DiGraph(self._nodes)
        plt.figure(figsize=(6, 6))
        nx.draw(
            G,
            with_labels=True,
            node_color=self._colors,
            node_size=800,
            font_color="white",
            font_weight="bold",
            arrowsize=20,
        )
        plt.show()

    def __str__(self) -> str:
        if not self.nodes:
            return "Grafo Vazio"

        lines = ["Graph:"]
        for k, values in self.nodes.items():
            cor = (
                self.colors[k]
                if isinstance(k, int) and k < len(self.colors)
                else "Sem cor"
            )
            lines.append(
                f"  {k} -> {values}: Cor({', '.join(f'{x:.2f}' for x in cor)})"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        num_nodes = len(self.nodes)
        num_colors = len(self.colors)
        return f"Graph(num_nodes={num_nodes}, num_colors={num_colors})"


# implementado
def random_graph_matrix(n: int) -> np.ndarray:
    if n > 100:
        print("Não foi testado para valores maiores que 100, tente valores menores")
        return np.array([[0]])
    random_matrix = np.random.randint(2, size=(n, n))
    np.fill_diagonal(random_matrix, 0)
    return random_matrix


# nsei
def random_graph_edge_list(n: int) -> np.ndarray:
    m = random_graph_matrix(n)
    u, v = [], []
    for i, node in enumerate(m):
        for j, link in enumerate(node):
            if link != 0:
                u.append(i)
                v.append(j)

    return np.array([u, v])


# implementado
def get_graph_dict(graph: np.ndarray) -> GraphStructure:
    dict_graph = {i: [] for i in range(len(graph))}
    for i, node in enumerate(graph.tolist()):
        for j, link in enumerate(node):
            if link != 0:
                dict_graph[i].append(j)
    return dict(dict_graph)


# implementado
def get_graph_ndarray(graph: GraphStructure) -> np.ndarray:
    u, v = [], []
    for node, links in graph.items():
        for link in links:
            u.append(node)
            v.append(link)
    return np.array([u, v])


# implementado
def get_graph_matrix(graph: GraphStructure) -> list[list]:
    if not graph:
        return []
    n = max(graph.keys()) + 1
    m = [[0] * n for _ in range(n)]
    for k, values in graph.items():
        for v in values:
            m[k][v] = 1
    return m


def map_dfs_edges(
    graph: GraphStructure, start: Any, function: Callable[[Any, Any], Any]
) -> list:
    seen = set()
    operation_result = []

    def dfs(node):
        seen.add(node)
        for neighbor in graph.get(node, []):
            operation_result.append(function(node, neighbor))
            if neighbor not in seen:
                seen.add(node)
                dfs(neighbor)

    dfs(start)
    return operation_result


def map_all_edges(graph: GraphStructure, function: Callable[[Any, Any], Any]) -> list:
    result = []
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            result.append(function(node, neighbor))
    return result


def find_path(graph: dict, start: Any, end: Any, path=None):
    """
    https://www.python.org/doc/essays/graphs/
    """
    if not path:
        path = []
    path = path + [start]
    if start == end:
        return path
    if not graph.get(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath
    return None


def find_all_paths(graph: dict, start: Any, end: Any, path=None):
    """
    https://www.python.org/doc/essays/graphs/
    """

    if not path:
        path = []
    path = path + [start]
    if start == end:
        return [path]
    if not graph.get(start):
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_shortest_path(graph: dict, start: Any, end: Any, path=None):
    """
    https://www.python.org/doc/essays/graphs/
    """
    if not path:
        path = []
    path = path + [start]
    if start == end:
        return path
    if not graph.get(start):
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest
