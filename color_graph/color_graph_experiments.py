from color_graph.color_graph_network import (
    single_example_analysis,
    train_test_color_graph_model,
)
from color_graph.dataset_generator import (
    make_torch_dataset,
)
from color_graph.graph_utils import Graph


def main():
    graph = Graph.random(5, 0.5)
    train_dataset, test_dataset = make_torch_dataset(graph, 1000, 800)
    model = train_test_color_graph_model(
        (train_dataset, test_dataset), len(graph.nodes)
    )

    single_example_analysis(graph, model)
    graph.show()


if __name__ == "__main__":
    main()
