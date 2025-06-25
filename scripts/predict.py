import os
import json
import torch
import random
import argparse
import warnings
import omegaconf
import sympy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import defaultdict

from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, NNEquation, BFGSParams

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, default='/home/cezary/Projects/IEiAI/test_set/nc.csv')
    parser.add_argument('--output_dir', '-o', type=str, default='/home/cezary/Projects/IEiAI/results/')
    parser.add_argument('--model_ckpt', '-m', type=str, default='/home/cezary/Projects/IEiAI/weights/100M.ckpt')
    parser.add_argument('--model_conf', '-c', type=str, default='/home/cezary/Projects/IEiAI/jupyter/100M/config.yaml')
    parser.add_argument('--eq_setting', '-e', type=str, default='/home/cezary/Projects/IEiAI/jupyter/100M/eq_setting.json')
    parser.add_argument('--num_equations', '-n', type=int, default=None)
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_points(equation, support, n_points, eq_setting, n_trials=20):
    n_variables = len(list(eq_setting['total_variables']))
    _min, _max = support['min'], support['max']
    _eq = sp.lambdify(
        ','.join(eq_setting['total_variables']),
        equation,
        modules=[
            'numpy',
            {'ln': np.log, 'asin': np.arcsin, 'acos': np.arccos, 'tan': np.tan},
        ],
    )

    trial = 0
    points = torch.empty(size=(0,), dtype=torch.float)
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
        id2word={int(k): v for k, v in eq_setting["id2word"].items()},
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

    num_equations = len(data) if args.num_equations is None else \
        min(args.num_equations, len(data))
    preds = defaultdict(list)
    for i, _, expr, supp, num_points in tqdm(
        data[:num_equations].itertuples(), total=num_equations
    ):
        supp = eval(supp)
        X, y = sample_points(expr, supp['x_1'], num_points, eq_setting)
        if X is None or y is None:
            continue

        pred = fitfunc(X, y)[0]['output']
        expr_pred = pred['best_bfgs_preds'][0] if pred['best_bfgs_preds'] else ''
        preds['expr_true'].append(expr)
        preds['expr_pred'].append(expr_pred)
        preds['support'].append(supp)
        preds['variables'].append(list(supp.keys()))
        preds['num_points'].append(num_points)

        print('True:', expr)
        print('Pred:', expr_pred if expr_pred else 'error')

    return pd.DataFrame(preds)


def main():
    args = parse_args()
    set_seed()

    preds = predict(args)
    print(preds)
    print(f'Successful prediction for {len(preds)} out of {args.num_equations} equations')

    os.makedirs(args.output_dir, exist_ok=True)
    num_equations = args.num_equations if args.num_equations else 'all'
    preds.to_csv(os.path.join(args.output_dir, f'pred_nesymres_{num_equations}.csv'), index=False)


if __name__ == '__main__':
    main()
