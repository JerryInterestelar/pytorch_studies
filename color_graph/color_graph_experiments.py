from color_graph.color_graph_network import (
    ColorGraphNetwork,
    train_test_color_graph_model,
    eval_color_graph_input,
)
from color_graph.dataset_generator import (
    ColorGraphDataset,
    make_random_graph_dataset,
)
from color_graph.graph_utils import Graph
from color_graph.graph_generator import gen_color_dataset, squeese_dataset


def single_example_analysis(graph: Graph, graph_model: ColorGraphNetwork):
    n_samples = 100
    print(
        f"* Gerando um dataset simples de {n_samples} amostras para o calculo de acurácia"
    )
    graph_random_colors_dataset = ColorGraphDataset(
        squeese_dataset(gen_color_dataset(graph, n_samples))
    )
    print(f"* Mostrando a acurácia para o grafo: {graph.nodes}")
    accuracy = eval_color_graph_input(graph_random_colors_dataset, graph_model)
    print(
        f"A acurácia calculada para a avaliação das permutações de cores geradas pelo o dataset de {n_samples} amostras é de: {accuracy:.2f}%"
    )


if __name__ == "__main__":
    graph, train_dataset, test_dataset = make_random_graph_dataset(5, 0.5, 1000, 800)
    model = train_test_color_graph_model(
        (train_dataset, test_dataset), len(graph.nodes)
    )

    single_example_analysis(graph, model)
    graph.show()
