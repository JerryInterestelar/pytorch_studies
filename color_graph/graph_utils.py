import numpy as np
from collections import defaultdict


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
