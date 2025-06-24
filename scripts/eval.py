import os
import json
import torch
import random
import argparse
import omegaconf
import sympy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import defaultdict


from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, NNEquation, BFGSParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/test_set/nc.csv')
    parser.add_argument('--output_dir', '-o', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/test_results/')
    
    # parser.add_argument('--model_ckpt', '-m', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/weights/10M.ckpt')
    # parser.add_argument('--model_conf', '-c', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/jupyter/10MPaper/config.yaml')
    # parser.add_argument('--eq_setting', '-e', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/jupyter/10MPaper/equation_config.json')

    parser.add_argument('--model_ckpt', '-m', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/weights/100M.ckpt')
    parser.add_argument('--model_conf', '-c', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/jupyter/100M/config.yaml')
    parser.add_argument('--eq_setting', '-e', type=str, default='/home/cezary/Projects/IEiAI-NeSymReS/jupyter/100M/eq_setting.json')

    parser.add_argument('--num_points_pred', type=int, default=500)
    parser.add_argument('--num_points_eval', type=int, default=10000)
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_points(equation, support, n_points, eq_setting, n_trials=10):
    n_variables = len(list(eq_setting['total_variables']))
    _min, _max = support['min'], support['max']
    _eq = sp.lambdify(','.join(eq_setting['total_variables']), equation, modules='numpy')

    trial = 0
    # points = torch.empty(dtype=torch.float)
    points = torch.tensor([], dtype=torch.float)
    targets = points.clone()
    while trial < n_trials and len(points) < n_points:
        _points = torch.rand(n_points, n_variables) * (_max - _min) + _min
        _targets = _eq(**{
            var: _points[:, i].cpu()
            for i, var in enumerate(eq_setting['total_variables'])
        })
        mask = torch.logical_and(~_targets.isnan(), ~_targets.isinf())
        points = torch.cat([points, _points[mask]])
        targets = torch.cat([targets, _targets[mask]])

    if len(points) >= n_points:
        return points[:n_points], targets[:n_points]
    else:
        return None, None


def predict(args):
    data = pd.read_csv(args.input_file)
    print(data)

    with open(args.eq_setting, 'r') as fp:
        eq_setting = json.load(fp)

    cfg = omegaconf.OmegaConf.load(args.model_conf)
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
        id2word={int(k): v for k,v in eq_setting["id2word"].items()},
        una_ops=eq_setting["una_ops"],
        bin_ops=eq_setting["bin_ops"],
        total_variables=list(eq_setting["total_variables"]),
        total_coefficients=list(eq_setting["total_coefficients"]),
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=cfg.inference.beam_size,
    )

    model = Model.load_from_checkpoint(args.model_ckpt, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    preds = defaultdict(list)

    for i, _, expr, supp, num_points in tqdm(
        data[:10].itertuples(), total=len(data)
    ):
        _supp = eval(supp)['x_1']
        X, y = sample_points(expr, _supp, num_points, eq_setting)
        # eq_true = sp.sympify(expr)
        if X is None or y is None:
            continue

        pred = fitfunc(X, y)

        preds['expr_true'].append(expr)
        preds['expr_pred'].append(pred['best_bfgs_preds'][0])
        preds['support'].append(supp)
        preds['num_points'].append(num_points)
        print(expr)
        print(pred['best_bfgs_preds'][0])

    return pd.DataFrame(preds)

def evaluate():
    pass


def main():
    args = parse_args()
    set_seed()

    preds = predict(args)
    print(preds)


if __name__ == '__main__':
    main()
