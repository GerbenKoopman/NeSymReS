import pandas as pd
import numpy as np
import ast
from sympy import symbols, lambdify, sympify

df = pd.read_csv("test_set/nc.csv")

def generate_dataset(row):
    expr_str = row["eq"]
    support = ast.literal_eval(row["support"])
    num_points = row["num_points"]
    
    # Get sorted variable names
    variables = sorted(support.keys())
    sym_vars = symbols(variables)

    # Sample input X
    X = np.stack([
        np.random.uniform(support[var]["min"], support[var]["max"], size=num_points)
        for var in variables
    ], axis=1)

    # Convert expression to function
    expr = sympify(expr_str)
    func = lambdify(sym_vars, expr, modules="numpy")

    X_list, y_list = [], []
    while sum(len(a) for a in y_list) < num_points:
        batch = max(num_points, 1000)
        Xb = np.stack([
            np.random.uniform(support[var]["min"], support[var]["max"], size=batch)
            for var in variables
        ], axis=1)
        yb = func(*[Xb[:, i] for i in range(Xb.shape[1])])
        mask = np.isfinite(yb)
        X_list.append(Xb[mask])
        y_list.append(yb[mask])

    X_all = np.concatenate(X_list)[:num_points]
    y_all = np.concatenate(y_list)[:num_points]
    
    return X_all, y_all, expr_str   
