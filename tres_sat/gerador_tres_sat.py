import random
import pandas as pd

type Expression = list[dict[int, bool]]


def generate_variables(n_variables: int) -> dict[int, bool]:
    return {i: random.choice([True, False]) for i in range(n_variables)}


def tree_sat_expression(n_variables: int, n_clauses: int) -> Expression:
    clauses: Expression = []

    for _ in range(n_clauses):
        indices_trio = sorted(random.sample(range(n_variables), 3))
        clauses.append({i: random.choice([True, False]) for i in indices_trio})

    return clauses


def eval_tree_sat(variables: dict[int, bool], expression: Expression) -> bool:
    analyzed_clauses = []
    for clause in expression:
        analyzed_variables = [
            variables[i] if value else not variables[i] for i, value in clause.items()
        ]
        analyzed_clauses.append(any(analyzed_variables))
    return all(analyzed_clauses)


def print_expression(expression: Expression) -> None:
    formatted_clauses = []

    for clause in expression:
        literals = [f"x{i}" if v else f"!x{i}" for i, v in clause.items()]

        formatted_clauses.append(f"({' or '.join(literals)})")

    expression_string = " and ".join(formatted_clauses)
    print(expression_string)


def generate_dataset(n_variables: int, n_clauses: int, file_name: str):
    tree_sat_list = tree_sat_expression(n_variables, n_clauses)
    print_expression(tree_sat_list)
    lines = []
    for _ in range(1000):
        x = generate_variables(n_variables)
        line = [1 if value else 0 for value in x.values()]
        line.append(1 if eval_tree_sat(x, tree_sat_list) else 0)
        lines.append(line)
    df = pd.DataFrame(lines)
    print(df)
    df.to_csv(file_name, index=False, header=True)


if __name__ == "__main__":
    generate_dataset(8, 10, "tres_sat_dataset_8_10.csv")
