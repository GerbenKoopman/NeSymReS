import yaml
import pandas as pd

from scripts.run_dsr import run_dsr, run_dsr_noise
from scripts.bench_utils import generate_dataset, generate_dataset_test


def bench_dsr(path_to_test_set):
    df = pd.read_csv(path_to_test_set)

    methods = [
        ("dsr",       run_dsr),
    ]

    accuracy_records = []
    detailed_results = {}

    # Loop for running each method
    for name, fn in methods:
        per_eq = {}
        correct_flags = []

        for _, row in df.iterrows():
            X_train, y_train, _ = generate_dataset(row)
            X_test, y_test, _ = generate_dataset_test(row)

            mse, pred_expr, correct = fn(X_train, y_train, X_test, y_test)

            per_eq[row["eq"]] = {
                "mse":            mse,
                "predicted_expr": pred_expr,
                "correct":        correct
            }
            correct_flags.append(correct)

        # Calculate A_1 score from reported correct predictions
        accuracy = sum(correct_flags) / len(correct_flags)

        accuracy_records.append({
            "method":   name,
            "accuracy": accuracy
        })
        detailed_results[name] = per_eq

    out_df = pd.DataFrame(accuracy_records)
    out_df.to_csv("benchmark_accuracies.csv", index=False)

    return detailed_results



def bench_dsr_noise(path_to_test_set):
    df = pd.read_csv(path_to_test_set)

    methods = [
        ("dsr",       run_dsr_noise),
    ]

    accuracy_records = []
    detailed_results = {}

    for name, fn in methods:
        per_eq = {}
        correct_flags = []

        for _, row in df.iterrows():
            X_train, y_train, _ = generate_dataset(row)
            X_test, y_test, _ = generate_dataset_test(row)

            mse, pred_expr, correct = fn(X_train, y_train, X_test, y_test)

            per_eq[row["eq"]] = {
                "mse":            mse,
                "predicted_expr": pred_expr,
                "correct":        correct
            }
            correct_flags.append(correct)


        accuracy = sum(correct_flags) / len(correct_flags)

        accuracy_records.append({
            "method":   name,
            "accuracy": accuracy
        })
        detailed_results[name] = per_eq

    out_df = pd.DataFrame(accuracy_records)
    out_df.to_csv("output/benchmark_accuracies_noise.csv", index=False)

    #print(detailed_results)

    return detailed_results
