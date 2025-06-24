import sys
import os
import torch
from pathlib import Path
from functools import partial
from sympy import lambdify
import json
import omegaconf

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

    number_of_points = 500
    n_variables = 2
    max_supp = cfg.dataset_train.fun_support["max"]
    min_supp = cfg.dataset_train.fun_support["min"]
    X = torch.rand(number_of_points, len(list(eq_setting["total_variables"])))*(max_supp-min_supp)+min_supp
    X[:, n_variables:] = 0
    target_eq = "x_1*sin(x_1)+cos(x_2)"  
    X_dict = {x: X[:, idx].cpu() for idx, x in enumerate(eq_setting["total_variables"])}
    y = lambdify(",".join(eq_setting["total_variables"]), target_eq)(**X_dict)

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    output = fitfunc(X, y)
    print(output)

if __name__ == "__main__":
    main()