from color_graph.color_graph_network import (
    single_example_analysis,
    train_test_color_graph_model,
)
from color_graph.dataset_generator import (
    make_torch_dataset,
)
from color_graph.graph_utils import Graph


def main():
    graph = Graph.random(14, 0.1)
    train_dataset, test_dataset, real_dists = make_torch_dataset(graph, 10000, 8000)
    model = train_test_color_graph_model(
        (train_dataset, test_dataset), len(graph.nodes)
    )

    print(f"Distribuição de resultados reais (válido x inválido):  {real_dists}")
    print(f"Distribuição do dataset de treino:  {train_dataset.y_distribution()}")
    print(f"Distribuição do dataset de teste:  {test_dataset.y_distribution()}")
    single_example_analysis(graph, model)
    graph.show()


if __name__ == "__main__":
    main()
