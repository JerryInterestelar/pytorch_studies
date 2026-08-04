import torch
from torch import Tensor

from tree_sat.tree_sat_network import TreeSATNetwork
from tree_sat.dataset_generator import TreeSATDataset
from core.metrics import binary_accuracy

n_input = 10
n_clauses = 60

tree_sat_dataset = TreeSATDataset(
    f"./data/datasets/test_tree_sat_dataset_{n_input}_{n_clauses}.csv"
)

TREE_SAT_MODEL_FILE = f"./data/models/tree_sat_model_{n_input}x{n_clauses}_weights.pth"


weights = torch.load(TREE_SAT_MODEL_FILE, weights_only=True)

model = TreeSATNetwork(n_input, 1)
model.load_state_dict(weights)


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


if __name__ == "__main__":
    print(
        f"A acurária do dataset de {n_input} inputs para o seu modelo treinado com {n_clauses} cláusulas é de: {100 * eval_tree_sat_input(tree_sat_dataset, model)}%"
    )
