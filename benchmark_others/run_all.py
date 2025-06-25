import yaml
import pandas as pd

from scripts.run_gp import run_gp, run_gp_noise
from scripts.run_sklearngp import run_sklearngp, run_sklearngp_noise
from scripts.bench_utils import generate_dataset


def run_all(path_to_test_set):
    df = pd.read_csv(path_to_test_set)

    gp_cfg    = yaml.safe_load(open("benchmark_others/configs/gplearn.yaml"))
    skgp_cfg  = yaml.safe_load(open("benchmark_others/configs/sklearngp.yaml"))

    methods = [
        ("gplearn",       run_gp,        gp_cfg),
        ("sklearngp",     run_sklearngp, skgp_cfg),
    ]

    accuracy_records = []
    detailed_results = {}

    for name, fn, cfg in methods:
        per_eq = {}
        correct_flags = []

        for _, row in df.iterrows():
            X, y, _ = generate_dataset(row)

            mse, pred_expr, correct = fn(X, y, cfg)

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
    out_df.to_csv("benchmark_accuracies.csv", index=False)

    return detailed_results


def run_all_noise(path_to_test_set):
    df = pd.read_csv(path_to_test_set)

    gp_cfg    = yaml.safe_load(open("benchmark_others/configs/gplearn.yaml"))
    skgp_cfg  = yaml.safe_load(open("benchmark_others/configs/sklearngp.yaml"))

    methods = [
        ("gplearn",       run_gp_noise,        gp_cfg),
        ("sklearngp",     run_sklearngp_noise, skgp_cfg),
    ]

    accuracy_records = []
    detailed_results = {}

    for name, fn, cfg in methods:
        per_eq = {}
        correct_flags = []

        for _, row in df.iterrows():
            X, y, _ = generate_dataset(row)

            mse, pred_expr, correct = fn(X, y, cfg)

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
    out_df.to_csv("benchmark_accuracies.csv", index=False)

    return detailed_results

run_all("test_set/nc_small.csv")