import json
import pandas as pd

from scripts.run_dsr import run_dsr, run_dsr_noise
from scripts.bench_utils import generate_dataset, generate_dataset_test


def bench_dsr(path_to_test_set):
    df = pd.read_csv(path_to_test_set)

    with open("benchmark_others/configs/dsr.json") as f:
        cfg = json.load(f)
    

    accuracy_records = []
    detailed_results = {}

    per_eq = {}
    correct_flags = []

    for _, row in df.iterrows():
        X_train, y_train, _ = generate_dataset(row)
        X_test, y_test, _ = generate_dataset_test(row)

        mse, pred_expr, correct = run_dsr(X_train, y_train, X_test, y_test, cfg)

        per_eq[row["eq"]] = {
            "mse":            mse,
            "predicted_expr": pred_expr,
            "correct":        correct
        }
        correct_flags.append(correct)


    accuracy = sum(correct_flags) / len(correct_flags)

    accuracy_records.append({
        "method":   "dsr",
        "accuracy": accuracy
    })
    detailed_results["dsr"] = per_eq

    out_df = pd.DataFrame(accuracy_records)
    out_df.to_csv("output/benchmark_accuracies_dsr.csv", index=False)


    return detailed_results



def bench_dsr_noise(path_to_test_set):
    df = pd.read_csv(path_to_test_set)

    with open("benchmark_others/configs/dsr.json") as f:
        cfg = json.load(f)


    accuracy_records = []
    detailed_results = {}

    per_eq = {}
    correct_flags = []

    for _, row in df.iterrows():
        X_train, y_train, _ = generate_dataset(row)
        X_test, y_test, _ = generate_dataset_test(row)

        mse, pred_expr, correct = run_dsr_noise(X_train, y_train, X_test, y_test, cfg)

        per_eq[row["eq"]] = {
            "mse":            mse,
            "predicted_expr": pred_expr,
            "correct":        correct
        }
        correct_flags.append(correct)


    accuracy = sum(correct_flags) / len(correct_flags)

    accuracy_records.append({
        "method":   "dsr",
        "accuracy": accuracy
    })
    detailed_results["dsr"] = per_eq

    out_df = pd.DataFrame(accuracy_records)
    out_df.to_csv("output/benchmark_accuracies_dsr_noise.csv", index=False)


    return detailed_results


if __name__ == "__main__":
    path_to_test_set = "test_set/nc.csv"
    results = bench_dsr(path_to_test_set)
    print(results)

    results_noise = bench_dsr_noise(path_to_test_set)
    print(results_noise)