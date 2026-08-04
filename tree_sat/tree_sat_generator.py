import random
from itertools import product

import pandas as pd

type Expression = list[tuple[tuple[int, int], ...]]


def generate_random_variables(n_variables: int) -> list[int]:
    return [int(random.choice([True, False])) for i in range(n_variables)]


def tree_sat_expression(n_variables: int, n_clauses: int) -> Expression:
    clauses = set()

    while len(clauses) < n_clauses:
        indices_trio = sorted(random.sample(range(n_variables), 3))
        clause = [tuple([i, random.choice([0, 1])]) for i in indices_trio]

        clauses.add(tuple(clause))
    return list(clauses)


def eval_tree_sat(variables: list[int], expression: Expression) -> int:
    analyzed_clauses = []
    for clause in expression:
        analyzed_variables = [
            variables[i] if value else not variables[i] for i, value in clause
        ]
        analyzed_clauses.append(any(analyzed_variables))
    return int(all(analyzed_clauses))


def print_expression(expression: Expression) -> None:
    formatted_clauses = []

    for clause in expression:
        literals = [f"x{i}" if v else f"!x{i}" for i, v in clause]

        formatted_clauses.append(f"({' or '.join(literals)})")

    expression_string = " and ".join(formatted_clauses)
    print(expression_string)


def generate_datasets_20_plus(
    n_variables: int, n_clauses: int, train_file_name: str, test_file_name: str
):
    tree_sat_list = tree_sat_expression(n_variables, n_clauses)
    print_expression(tree_sat_list)

    unique_cases = set()

    while len(unique_cases) < 1200:
        variables = random.choices([0, 1], k=n_variables)
        unique_cases.add(tuple(variables))
    full_lines = []
    for line in unique_cases:
        list_line = list(line)
        list_line.append(eval_tree_sat(list_line, tree_sat_list))
        full_lines.append(list_line)

    df = pd.DataFrame(full_lines)

    df[:1000].to_csv(train_file_name, index=False, header=False)
    df[1000:].to_csv(test_file_name, index=False, header=False)


def generate_datasets_20_less(
    n_variables: int, n_clauses: int, train_file_name: str, test_file_name: str
):
    tree_sat_list = tree_sat_expression(n_variables, n_clauses)
    print_expression(tree_sat_list)

    unique_cases = list(product([0, 1], repeat=n_variables))
    random.shuffle(unique_cases)

    full_lines = []
    for line in unique_cases:
        list_line = list(line)
        list_line.append(eval_tree_sat(list_line, tree_sat_list))
        full_lines.append(list_line)

    df = pd.DataFrame(full_lines)
    print(f"Incidência de valores 1 e 0: {df.iloc[:, -1].value_counts()}")
    df[:800].to_csv(train_file_name, index=False, header=False)
    df[800:].to_csv(test_file_name, index=False, header=False)


if __name__ == "__main__":
    n_variables = 10
    n_clauses = 20
    generate_datasets_20_less(
        n_variables,
        n_clauses,
        f"./data/datasets/train_tree_sat_dataset_{n_variables}_{n_clauses}.csv",
        f"./data/datasets/test_tree_sat_dataset_{n_variables}_{n_clauses}.csv",
    )
    # generate_dataset(43, 10, "./data/datasets/tres_sat_dataset_43_10.csv")
    # generate_dataset(60, 10, "./data/datasets/tres_sat_dataset_60_10.csv")
