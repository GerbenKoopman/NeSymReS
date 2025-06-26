import pandas as pd
import numpy as np
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

def extract_variables(expr):
    # Extract variable names from a single expression
    try:
        return [str(s) for s in sp.sympify(expr).free_symbols]
    except Exception as e:
        print(f"Failed to parse expression '{expr}': {e}")
        return []

def safe_eval(expr, subs):
    try:
        return float(expr.evalf(subs=subs))
    except Exception:
        return np.nan

def sample_points(support, num_points, variables, ood=False):
    lo, hi = support['min'], support['max']
    if ood:
        length = hi - lo
        lo_ood = lo - length
        hi_ood = hi + length
        return np.random.uniform(lo_ood, hi_ood, size=(num_points, len(variables)))
    else:
        return np.random.uniform(lo, hi, size=(num_points, len(variables)))

def evaluate_one(expr_true, expr_pred, support, variables, num_points=128, ood=False):
    X = sample_points(support, num_points, variables, ood=ood)
    if X is None or len(X) == 0:
        return None, None, None
    try:
        # Use lambdify for fast batch evaluation
        f_true = sp.lambdify(variables, sp.sympify(expr_true), modules='numpy')
        f_pred = sp.lambdify(variables, sp.sympify(expr_pred), modules='numpy')
    except Exception as e:
        print(f"Failed to parse expression: {e}")
        return None, None, None
    try:
        # X shape: (num_points, n_vars), need to unpack as columns
        y_true = f_true(*X.T)
        y_pred = f_pred(*X.T)
    except Exception as e:
        print(f"Failed to evaluate: {e}")
        return None, None, None
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    if np.sum(mask) < 2:
        return None, None, None
    y_true, y_pred = y_true[mask], y_pred[mask]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    is_exact = str(expr_true).replace(' ', '') == str(expr_pred).replace(' ', '')
    return mse, r2, is_exact

def main(csv_path):
    data = pd.read_csv(csv_path)
    n_total = len(data)
    iid_mse_list, iid_r2_list, iid_exact_list = [], [], []
    ood_mse_list, ood_r2_list, ood_exact_list = [], [], []

    # for idx, row in data.iterrows():
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
        expr_true = row['expr_true']
        expr_pred = row['expr_pred']
        support = eval(row['support'])
        # Extract variables from both expressions and merge
        variables = sorted(list(
            set(extract_variables(expr_true)) | set(extract_variables(expr_pred))
        ))

        # In-distribution
        mse_iid, r2_iid, exact_iid = evaluate_one(expr_true, expr_pred, support, variables, num_points=128, ood=False)
        if mse_iid is not None:
            iid_mse_list.append(mse_iid)
            iid_r2_list.append(r2_iid)
            iid_exact_list.append(exact_iid)
        else:
            print(f"Row {idx} IID: too few valid points, skipped.")

        # Out-of-distribution
        mse_ood, r2_ood, exact_ood = evaluate_one(expr_true, expr_pred, support, variables, num_points=128, ood=True)
        if mse_ood is not None:
            ood_mse_list.append(mse_ood)
            ood_r2_list.append(r2_ood)
            # ood_exact_list.append(ood_exact_list)
            ood_exact_list.append(exact_ood)
        else:
            print(f"Row {idx} OOD: too few valid points, skipped.")

    # Summary
    print("\n==== In-Distribution (Aiid) ====")
    print(f"Aiid (Top-1 Accuracy): {np.mean(iid_exact_list):.4f}")
    print(f"Mean MSE: {np.mean(iid_mse_list):.6f}")
    print(f"Mean R²: {np.mean(iid_r2_list):.6f}")

    print("\n==== Out-of-Distribution (Aood) ====")
    print(f"Aood (Top-1 Accuracy): {np.mean(ood_exact_list):.4f}")
    print(f"Mean MSE: {np.mean(ood_mse_list):.6f}")
    print(f"Mean R²: {np.mean(ood_r2_list):.6f}")

    print("\n==== Overall (A1) ====")
    all_exact = iid_exact_list + ood_exact_list
    print(f"A1 (Top-1 Accuracy): {np.mean(all_exact):.4f}")

if __name__ == "__main__":
    main("/home/scur1229/nesymres/test_results/pred_nesymres.csv")