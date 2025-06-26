import yaml
import pandas as pd
from tqdm import tqdm
import os
import sympy as sp

from scripts.run_gp import run_gp, run_gp_noise
from scripts.run_sklearngp import run_sklearngp, run_sklearngp_noise
from scripts.bench_utils import generate_dataset, generate_dataset_test

os.makedirs("output", exist_ok=True)

converter = {
    "add": sp.Add,
    "mul": sp.Mul,
    "sub": lambda x, y: x - y,
    "div": lambda x, y: x / y,
    "sqrt": sp.sqrt,
    "tan": sp.tan, 
    "log": sp.ln,
    "neg": lambda x: -x,
    "inv": lambda x: 1 / x,
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
}

def convert_gplearn_expr(expr_pred):
    if not expr_pred:
        return ""
    try:
        eq_gplearn = sp.sympify(expr_pred, locals=converter)
        expr_pred_str = (
            str(eq_gplearn)
            .replace('X0', 'x_0')
            .replace('X1', 'x_1')
            .replace('X2', 'x_2')
            .replace('X3', 'x_3')
        )
        return expr_pred_str
    except Exception as e:
        return ""  

def bench_all(path_to_test_set):
    df = pd.read_csv(path_to_test_set)
    df = df.head(100)

    with open("/home/scur1229/nesymres/benchmark_others/configs/gplearn.yaml") as f:
        gp_cfg = yaml.safe_load(f)
    with open("/home/scur1229/nesymres/benchmark_others/configs/sklearngp.yaml") as f:
        skgp_cfg = yaml.safe_load(f)

    methods = [
        ("gplearn", run_gp, gp_cfg),
        ("sklearngp", run_sklearngp, skgp_cfg),
    ]

    all_results = []

    for name, fn, cfg in methods:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {name}"):
            X_train, y_train, _ = generate_dataset(row)
            X_test, y_test, _ = generate_dataset_test(row)
            mse, pred_expr, correct = fn(X_train, y_train, X_test, y_test, cfg)

            if name == "gplearn":
                expr_pred_str = convert_gplearn_expr(pred_expr)
            else:
                expr_pred_str = pred_expr if pred_expr is not None else ""

            all_results.append({
                "index": row.name,
                "method": name,
                "expr_true": row["eq"],
                "expr_pred": expr_pred_str,
                "mse": mse,
                "correct": correct
            })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv("output/benchmark_accuracies.csv", index=False)

def bench_all_noise(path_to_test_set):
    df = pd.read_csv(path_to_test_set)
    df = df.head(100)

    with open("/home/scur1229/nesymres/benchmark_others/configs/gplearn.yaml") as f:
        gp_cfg = yaml.safe_load(f)
    with open("/home/scur1229/nesymres/benchmark_others/configs/sklearngp.yaml") as f:
        skgp_cfg = yaml.safe_load(f)

    methods = [
        ("gplearn", run_gp_noise, gp_cfg),
        ("sklearngp", run_sklearngp_noise, skgp_cfg),
    ]

    all_results = []

    for name, fn, cfg in methods:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {name} (noise)"):
            X_train, y_train, _ = generate_dataset(row)
            X_test, y_test, _ = generate_dataset_test(row)
            mse, pred_expr, correct = fn(X_train, y_train, X_test, y_test, cfg)

            if name == "gplearn":
                expr_pred_str = convert_gplearn_expr(pred_expr)
            else:
                expr_pred_str = pred_expr if pred_expr is not None else ""

            all_results.append({
                "index": row.name,
                "method": name,
                "expr_true": row["eq"],
                "expr_pred": expr_pred_str,
                "mse": mse,
                "correct": correct
            })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv("output/benchmark_accuracies_noise.csv", index=False)

if __name__ == "__main__":
    path_to_test_set = "/home/scur1229/nesymres/test_set/nc.csv"
    bench_all(path_to_test_set)
    # bench_all_noise(path_to_test_set)