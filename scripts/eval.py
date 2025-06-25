import json
import argparse
import numpy as np
import sympy as sp
import pandas as pd
import func_timeout
from tqdm import tqdm
from collections import defaultdict
from sympy.core import Number

from predict import sample_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/cezary/Projects/IEiAI/results/pred_nesymres_200.csv')
    parser.add_argument('--output_file', type=str, default='/home/cezary/Projects/IEiAI/results/metrics_nesymres_200.csv')
    parser.add_argument('--eq_setting', type=str, default='/home/cezary/Projects/IEiAI/jupyter/100M/eq_setting.json')
    parser.add_argument('--num_points', type=int, default=10000)
    return parser.parse_args()


def symbolic_match(exp1, exp2, rtol=1e-1, atol=2e-3, recursive_call=False):
    def _impl(exp1, exp2, rtol=1e-1, atol=2e-3, recursive_call=False):
        eq1, eq2 = sp.sympify(exp1), sp.sympify(exp2)

        # if sp.simplify(eq1 - eq2) == 0:
        #     return True
    
        reps = {
            i: i.round() for i in eq2.atoms(sp.core.Float)
            if np.isclose(float(i), float(i.round()), rtol=rtol, atol=atol)
        }
        return sp.simplify(eq1 - eq2.subs(reps)) == 0
    
    try:
        return func_timeout.func_timeout(
            20, _impl,
            kwargs=dict(exp1=exp1, exp2=exp2, rtol=rtol, atol=atol),
        )
    except func_timeout.FunctionTimedOut:
        pass
    return np.nan


def safe_func(f, max_wait, default_value):
    try:
        return func_timeout.func_timeout(max_wait, long_function)
    except func_timeout.FunctionTimedOut:
        pass
    return default_value



def evaluate(data, num_points, eq_setting):
    output = defaultdict(list)

    for i, row in tqdm(data.iterrows(), total=len(data)):
        eq_true = sp.sympify(row['expr_true'])
        eq_pred = sp.sympify(row['expr_pred'])
        X, y = sample_points(
            eq_true, {'max': 10, 'min': -10},
            num_points, eq_setting, n_trials=25,
        )
        if X is None or y is None:
            continue

        X, y = X.cpu().numpy(), y.cpu().numpy()
        y_pred = sp.lambdify(','.join(eq_setting['total_variables']), eq_pred, 'numpy')(*X.T)
        mask = ~np.isnan(y_pred) & ~np.isinf(y_pred)

        if mask.sum() == 0:
            continue

        y_true_valid = y[mask]
        y_pred_valid = y_pred[mask]

        # R²
        ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
        ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)

        results = {
            'expr_true': row['expr_true'],
            'expr_pred': row['expr_pred'],
            'mse': np.mean((y_true_valid - y_pred_valid) ** 2),
            'mae': np.mean(np.abs(y_true_valid - y_pred_valid)),
            'r2': 1 - (ss_res / ss_tot) if ss_tot > 0 else 0,
            'pointwise_acc': np.isclose(y[mask], y_pred[mask], rtol=0.05, atol=0.001).mean(),
            'symbolic_match': float(symbolic_match(eq_true, eq_pred)),
        }

        for k, v in results.items():
            output[k].append(v)

    return pd.DataFrame(output)

def main():
    args = parse_args()

    data = pd.read_csv(args.input_file)
    with open(args.eq_setting, 'r') as fp:
        eq_setting = json.load(fp)
    print('Input data:')
    print(data)
    results = evaluate(data, args.num_points, eq_setting)
    results.to_csv(args.output_file, index=False)

    with np.printoptions(precision=4, suppress=True, threshold=5):
        print(results)

    print('\nMetrics (means):')
    for i, row in results.mean(numeric_only=True).items():
        print(f'{i:>16}: {row:.4f}')
        # print(results.mean(numeric_only=True))


if __name__ == '__main__':
    main()
