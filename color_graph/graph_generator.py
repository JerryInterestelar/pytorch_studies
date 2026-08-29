from graph_utils import Graph


def main():
    n = 5
    graph = Graph.random(n)
    print(graph)
    result = graph.map_all_edges(lambda n, e: graph.colors[n] == graph.colors[e])
    print(result)
    print("Grafo mestre" if not any(result) else "Grafo comum")
    print(graph.find_path(0, 4))
    graph.show()
    #


if __name__ == "__main__":
    main()
