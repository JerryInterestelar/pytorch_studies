from graph_utils import get_graph_dict, random_graph_edge_list  # noqa: F401


def main():
    # np_graph = np.array([[0, 0, 1, 1, 2, 3, 3, 4, 5], [1, 2, 2, 3, 3, 2, 4, 5, 2]])
    # a = {0: [1, 2], 1: [2, 3], 2: [3], 3: [2, 4], 4: [5], 5: [2]}
    for n, e in random_graph_edge_list(6).T.tolist():
        print(f"{n} {e}")


if __name__ == "__main__":
    main()
