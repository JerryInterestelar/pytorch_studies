import torch
from torch import Tensor

from tree_sat.tree_sat_network import TreeSATNetwork

TREE_SAT_MODEL_FILE = "./data/models/tree_sat_model_8x1_weights.pth"


weights = torch.load(TREE_SAT_MODEL_FILE, weights_only=True)

model = TreeSATNetwork(8, 1)
model.load_state_dict(weights)


def eval_tree_sat_input(
    x: Tensor,
    model: TreeSATNetwork,
):
    model.eval()

    input_batch = x.unsqueeze(0)

    with torch.no_grad():
        predicion: Tensor = model(input_batch)
        predicted_class = (predicion >= 0.0).type(torch.float).item()
    return predicted_class


if __name__ == "__main__":
    x = torch.tensor([1, 0, 0, 1, 0, 1, 0, 0], dtype=torch.float)
    print(eval_tree_sat_input(x, model))
