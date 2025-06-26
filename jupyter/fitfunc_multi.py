import sys
import os
import torch
from pathlib import Path
from functools import partial
from sympy import lambdify, sympify
import json
import omegaconf
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src')))

from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams

def main():
    with open('100M/eq_setting.json', 'r') as json_file:
        eq_setting = json.load(json_file)
    cfg = omegaconf.OmegaConf.load("100M/config.yaml")

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(
        word2id=eq_setting["word2id"],
        id2word={int(k): v for k, v in eq_setting["id2word"].items()},
        una_ops=eq_setting["una_ops"],
        bin_ops=eq_setting["bin_ops"],
        total_variables=list(eq_setting["total_variables"]),
        total_coefficients=list(eq_setting["total_coefficients"]),
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=cfg.inference.beam_size
    )

    weights_path = "../weights/100M.ckpt"
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    from nesymres.architectures.beam_search import generate_multiple_beams
    beam_configs = [
        {"beam_size": 5, "length_penalty": 1.0, "max_len": 100},
        {"beam_size": 10, "length_penalty": 0.8, "max_len": 150},
        {"beam_size": 3, "length_penalty": 1.2, "max_len": 80},
    ]
    fitfunc = partial(model.fitfunc, cfg_params=params_fit, beam_configs=beam_configs)

    df = pd.read_csv("/home/scur1229/nesymres/test_set/nc.csv")
    total_variables = list(eq_setting["total_variables"])
    max_supp = cfg.dataset_train.fun_support["max"]
    min_supp = cfg.dataset_train.fun_support["min"]

    for idx, row in df.iterrows():
        if idx >= 3:
            break
        eq_str = row['eq']
        if "log" in eq_str:
            continue
        support = eval(row['support']) if isinstance(row['support'], str) else row['support']
        num_points = int(row['num_points'])

        X_np = []
        for var in total_variables:
            if var in support:
                var_min = support[var]['min']
                var_max = support[var]['max']
                points = np.random.uniform(var_min, var_max, num_points)
            else:
                points = np.zeros(num_points)
            X_np.append(points)
        X_np = np.stack(X_np, axis=1)  # shape: (num_points, num_total_variables)
        X = torch.from_numpy(X_np).float()

        X_dict = {x: X[:, idx].cpu().numpy() for idx, x in enumerate(total_variables)}
        try:
            expr = sympify(eq_str)
            func = lambdify(total_variables, expr, "numpy")
            y = func(**X_dict)
            if np.isscalar(y):
                y = np.full(num_points, y)
            y = torch.from_numpy(np.array(y)).float()
        except Exception as e:
            print(f"Equation{idx} is failed to fit: {e}")
            continue

        print(f"\nEquation {idx}: {eq_str}")
        print("X shape:", X.shape, "y shape:", y.shape)

        try:
            output = fitfunc(X, y)
            # print("Fit result:", output)
            for result in output:
                if "output" in result and "best_bfgs_preds" in result["output"]:
                    print("config:", result["config"])
                    print("best_bfgs_preds:", result["output"]["best_bfgs_preds"])
        except Exception as e:
            print(f"Fit failed: {e}")

if __name__ == "__main__":
    main()