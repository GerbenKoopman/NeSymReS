import os
import json
import pysr
import torch
import random
import argparse
import warnings
import omegaconf
import sympy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
# from dso import DeepSymbolicOptimizer
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
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
    parser.add_argument('--noise_stds', '-n', type=float, nargs='+', default=[0])
    parser.add_argument('--num_equations', '-N', type=int, default=None)
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


def predict(args, noise_std):
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

    all_X = np.empty(
        (num_equations, data[:num_equations].num_points.max(), 3),
        dtype=np.float64,
    )
    all_y = np.empty(
        (num_equations, data[:num_equations].num_points.max()),
        dtype=np.float64,
    )
    all_X[:], all_y[:] = np.nan, np.nan

    preds = defaultdict(list)
    for i, _, expr, supp, num_points in tqdm(
        data[:num_equations].itertuples(), total=num_equations
    ):
        supp = eval(supp)
        X, y = sample_points(expr, supp['x_1'], num_points, eq_setting)
        if X is None or y is None:
            continue

        if noise_std > 0:
            noise = np.random.normal(loc=0.0, scale=noise_std, size=y.shape)
            print('noise', noise[:10])
            print('y', y[:10])
            y += noise
            print('y', y[:10])

        all_X[i], all_y[i] = X, y

        # NeSymRes
        pred_nesymres = fitfunc(X, y)[0]['output']
        expr_nesymres = pred_nesymres['best_bfgs_preds'][0] \
            if pred_nesymres['best_bfgs_preds'] else ''

        def protected_exp(x):
            try:
                return np.where(x < 100, np.exp(x), 1e10)  # prevent overflow
            except:
                return np.ones_like(x) * 1e1

        exp_function = make_function(function = protected_exp, name='exp', arity=1)

        # GPLearn
        model_gp = SymbolicRegressor(
            population_size=4096,
            tournament_size=20,
            p_crossover=0.9,
            p_subtree_mutation=0.01,
            function_set=[
                "add", "sub", "mul", "div", "sqrt", "tan", 
                "log", "neg", "inv", "sin", "cos", exp_function
            ],
            const_range=(-4*np.pi, 4*np.pi),
            verbose=0,
        )

        # PySR
        model_pysr = pysr.PySRRegressor(
            niterations=50,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[
                "cos", "exp", "sin", "asin", "log","tan",
                "pow2(x) = x^2", "pow3(x) = x^3",
                "pow4(x) = x^4", "pow5(x) = x^5", "inv(x) = 1/x",
            ],
            extra_sympy_mappings={
                "inv": lambda x: 1 / x,
                "pow2": lambda x: x ** 2,
                "pow3": lambda x: x ** 3,
                "pow4": lambda x: x ** 4,
                "pow5": lambda x: x ** 5,
            },
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        )

        # DSO
        # model = DeepSymbolicOptimizer("path/to/config.json")

        model_gp.fit(X, y)
        model_pysr.fit(X, y)

        # print(model_pysr, type(model_pysr))
        # print(model_pysr.equations_.columns, type(model_pysr))
        # print(model_pysr.equations_)
        # print(vars(model_pysr))
        # print(model_pysr.equations_.iloc[-1])
        # print(model_pysr.equations_[model_pysr.equations_['pick'] == '>>>>'].equation)
        # print(model_pysr.
        # print(expr_gplearn._program.__str__())

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
        eq_gplearn = sp.sympify(model_gp._program.__str__(), locals=converter)
        expr_gplearn = (
            str(eq_gplearn)
            .replace('X0', 'x_0')
            .replace('X1', 'x_1')
            .replace('X2', 'x_2')
        )

        preds['expr_true'].append(expr)
        preds['expr_nesymres'].append(expr_nesymres)
        preds['expr_gplearn'].append(expr_gplearn)  # model_gp._program.__str__())
        preds['expr_pysr'].append(str(model_pysr.sympy()))
        preds['support'].append(supp)
        preds['variables'].append(list(supp.keys()))
        preds['num_points'].append(num_points)

        print('')
        print('    True:', expr)
        print('NeSymRes:', expr_nesymres if expr_nesymres else 'error')
        print(' GPLearn:', model_gp._program.__str__())
        print(' GPLearn:', expr_gplearn)
        print('    PySR:', str(model_pysr.sympy()))
        print('')

    return pd.DataFrame(preds), all_X, all_y


def main():
    args = parse_args()
    set_seed()

    for i, std in enumerate(args.noise_stds):
        print(f'\nnoise std = {std} ({i+1}/{len(args.noise_stds)})')
        preds, X, y = predict(args, std)
        print(preds)
        print(f'Successful prediction for {len(preds)} out of {args.num_equations} equations')

        os.makedirs(args.output_dir, exist_ok=True)
        num_equations = args.num_equations if args.num_equations else 'all'
        preds.to_csv(
            os.path.join(args.output_dir, 'noise', f'pred_nesymres_{num_equations}_{std:.5f}.csv'),
            index=False,
        )
        np.savez_compressed(f'point_{num_equations}_{std}.npz', X=X, y=y)


if __name__ == '__main__':
    main()
