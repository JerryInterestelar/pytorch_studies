import torch
from torch import Tensor

from tree_sat.tree_sat_network import TreeSATNetwork
from tree_sat.dataset_generator import TreeSATDataset
from tree_sat.tree_sat_generator import generate_datasets_20_less
from core.metrics import binary_accuracy


def eval_tree_sat_input(
    data_set: TreeSATDataset,
    model: TreeSATNetwork,
):
    model.eval()

    accuracy = 0
    with torch.no_grad():
        for x, y in data_set:
            prediction: Tensor = model(x)
            accuracy += binary_accuracy(prediction, y)
    return accuracy / (len(data_set))


def load_model(
    n_input: int,
    n_clauses: int,
) -> TreeSATNetwork:
    print(f"1 - Carregando o modelo {n_input}X{n_clauses} treinado com uma fórmula X")
    TREE_SAT_MODEL_FILE = (
        f"./data/models/tree_sat_model_{n_input}x{n_clauses}_weights.pth"
    )
    weights = torch.load(TREE_SAT_MODEL_FILE, weights_only=True)

    model = TreeSATNetwork(n_input, 1)
    model.load_state_dict(weights)
    return model


def test_new_input_and_formula_dataset():
    n_input = 10
    n_clauses = 5

    model = load_model(n_input, n_clauses)

    print("2 - Testando a acurária com o dataset de que treinou a rede com a fórmula X")
    model_dataset = TreeSATDataset.from_csv(
        f"./data/datasets/train_tree_sat_dataset_{n_input}_{n_clauses}.csv"
    )

    print("Fazendo o eval na rede e analizando a acurária")
    print(
        f"FORMULA X - A acurária do dataset de {n_input} inputs para o seu modelo treinado com {n_clauses} cláusulas é de: {100 * eval_tree_sat_input(model_dataset, model)}%"
    )

    print("3 - Testando a acurária com o dataset novo com uma fórmula Y")
    print(f"Gerando um novo dataset {n_input}X{n_clauses} com uma fórmula Y aleatória")
    new_formula_dataset = TreeSATDataset(generate_datasets_20_less(n_input, n_clauses))

    print("Fazendo o eval na rede e gerando a acurária")
    print(
        f"FORMULA Y - A acurária do dataset de {n_input} inputs para o seu modelo treinado com {n_clauses} cláusulas é de: {100 * eval_tree_sat_input(new_formula_dataset, model)}%"
    )


if __name__ == "__main__":
    test_new_input_and_formula_dataset()
