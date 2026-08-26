import numpy as np
from collections import defaultdict


def random_graph_matrix(n: int) -> np.ndarray:
    if n > 100:
        print("Não foi testado para valores maiores que 100, tente valores menores")
        return np.array([[0]])
    random_matrix = np.random.randint(2, size=(n, n))
    np.fill_diagonal(random_matrix, 0)
    return random_matrix


def random_graph_edge_list(n: int) -> np.ndarray:
    m = random_graph_matrix(n)
    u, v = [], []
    for i, node in enumerate(m):
        for j, link in enumerate(node):
            if link != 0:
                u.append(i)
                v.append(j)

    return np.array([u, v])


def get_graph_dict(graph: np.ndarray) -> dict:
    dict_graph = defaultdict(list)
    for u, v in graph.T.tolist():
        dict_graph[u].append(v)
    return dict(dict_graph)


def get_graph_ndarray(graph: dict) -> np.ndarray:
    u, v = [], []
    for node, links in graph.items():
        for link in links:
            u.append(node)
            v.append(link)
    return np.array([u, v])


def get_graph_matrix(graph: np.ndarray) -> list[list]:
    m = []
    n = len(set(graph[0]))
    for _ in range(n):
        m.append([0] * n)
    for n, e in graph.T.tolist():
        m[n][e] = 1
    return m


def find_path(graph: dict, start, end, path=[]):
    """
    https://www.python.org/doc/essays/graphs/
    """

    path = path + [start]
    if start == end:
        return path
    if not graph[start]:
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath
    return None


def find_all_paths(graph: dict, start, end, path=[]):
    """
    https://www.python.org/doc/essays/graphs/
    """

    path = path + [start]
    if start == end:
        return [path]
    if not graph[start]:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_shortest_path(graph: dict, start, end, path=[]):
    """
    https://www.python.org/doc/essays/graphs/
    """
    path = path + [start]
    if start == end:
        return path
    if not graph[start]:
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest
