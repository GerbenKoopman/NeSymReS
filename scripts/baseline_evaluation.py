#!/usr/bin/env python3
"""
Simple NESYMRES evaluation script that works with CSV test data.
This provides a baseline evaluation without requiring full HDF5 dataset setup.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nesymres.architectures.model import Model
from nesymres.dclasses import BFGSParams, FitParams, NNEquation
from nesymres.utils import load_metadata_hdf5
from nesymres import benchmark
import hydra
from omegaconf import DictConfig

def load_test_equations_from_csv(csv_path):
    """Load equations from CSV file format."""
    df = pd.read_csv(csv_path)
    equations = []
    
    for idx, row in df.iterrows():
        eq_data = {
            'idx': idx,
            'equation': row['eq'],
            'support': eval(row['support']) if isinstance(row['support'], str) else row['support'],
            'num_points': row['num_points']
        }
        equations.append(eq_data)
    
    return equations

def create_test_data_points(equation_str, support, num_points=500):
    """Create test data points for an equation."""
    variables = list(support.keys())
    
    # Create random points within the support
    X = []
    for var in variables:
        var_min = support[var]['min']
        var_max = support[var]['max']
        points = np.random.uniform(var_min, var_max, num_points)
        X.append(points)
    
    X = np.array(X).T  # Shape: (num_points, num_variables)
    
    # Evaluate the ground truth equation
    try:
        from sympy import lambdify, sympify
        expr = sympify(equation_str)
        func = lambdify(variables, expr, "numpy")
        
        if len(variables) == 1:
            y = func(X[:, 0])
        else:
            y = func(*X.T)
        
        # Handle scalar outputs
        if np.isscalar(y):
            y = np.full(num_points, y)
        
        return X, y
    except Exception as e:
        print(f"Error creating test data for equation '{equation_str}': {e}")
        return None, None

def evaluate_equation(model, equation_data, params_fit, beam_configs):
    """Evaluate a single equation."""
    print(f"Evaluating equation {equation_data['idx']}: {equation_data['equation']}")
    
    # Create test data
    X, y_true = create_test_data_points(
        equation_data['equation'],
        equation_data['support'],
        equation_data['num_points']
    )
    
    if X is None or y_true is None:
        return None
    
    # Remove NaN and inf values
    valid_mask = np.isfinite(y_true) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y_true = y_true[valid_mask]
    
    if len(X) < 10:  # Need minimum points for evaluation
        print(f"Too few valid points ({len(X)}) for equation {equation_data['idx']}")
        return None
    
    results = []
    
    # Test different beam configurations
    for beam_config in beam_configs:
        print(f"  Testing beam config: {beam_config}")
        
        # Update beam size in params
        params_fit.beam_size = beam_config['beam_size']
        
        try:
            # Create fitfunc with current parameters
            fitfunc = partial(model.fitfunc, cfg_params=params_fit)
            
            # Run prediction
            output = fitfunc(X.T, y_true)
            
            # Extract results
            predicted_eq = output.get('best_bfgs_preds', [''])[0] if output.get('best_bfgs_preds') else ''
            
            result = {
                'equation_idx': equation_data['idx'],
                'ground_truth': equation_data['equation'],
                'predicted': predicted_eq,
                'beam_config': beam_config,
                'num_valid_points': len(X),
                'output': output
            }
            
            # Simple accuracy metrics
            if predicted_eq:
                try:
                    # Evaluate predicted equation
                    from sympy import lambdify, sympify
                    variables = list(equation_data['support'].keys())
                    pred_expr = sympify(predicted_eq)
                    pred_func = lambdify(variables, pred_expr, "numpy")
                    
                    if len(variables) == 1:
                        y_pred = pred_func(X[:, 0])
                    else:
                        y_pred = pred_func(*X.T)
                    
                    if np.isscalar(y_pred):
                        y_pred = np.full(len(y_true), y_pred)
                    
                    # Calculate metrics
                    valid_pred_mask = np.isfinite(y_pred)
                    if np.sum(valid_pred_mask) > 0:
                        y_true_valid = y_true[valid_pred_mask]
                        y_pred_valid = y_pred[valid_pred_mask]
                        
                        mse = np.mean((y_true_valid - y_pred_valid) ** 2)
                        mae = np.mean(np.abs(y_true_valid - y_pred_valid))
                        
                        # R²
                        ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
                        ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        result.update({
                            'mse': float(mse),
                            'mae': float(mae),
                            'r2': float(r2),
                            'symbolic_match': str(predicted_eq).strip() == str(equation_data['equation']).strip()
                        })
                    
                except Exception as e:
                    print(f"    Error evaluating predicted equation: {e}")
                    result.update({
                        'mse': float('inf'),
                        'mae': float('inf'),
                        'r2': -float('inf'),
                        'symbolic_match': False,
                        'evaluation_error': str(e)
                    })
            
            results.append(result)
            print(f"    Result: {predicted_eq[:50]}{'...' if len(predicted_eq) > 50 else ''}")
            
        except Exception as e:
            print(f"    Error during prediction: {e}")
            results.append({
                'equation_idx': equation_data['idx'],
                'ground_truth': equation_data['equation'],
                'predicted': '',
                'beam_config': beam_config,
                'error': str(e)
            })
    
    return results

@hydra.main(config_path=".", config_name="config_wrapper")
def main(cfg: DictConfig):
    """Main evaluation function."""
    
    print("Starting NESYMRES Baseline Evaluation")
    print("=" * 50)
    
    # Load model
    model_path = Path(cfg.evaluation.checkpoint_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    print(f"Loading model from: {model_path}")
    
    # Use a dummy HDF5 metadata if available, or create minimal config
    try:
        # Try to load from existing validation data
        test_data = load_metadata_hdf5(Path("/home/gerben-koopman/studie/nesymres/data/validation"))
    except:
        print("No HDF5 validation data found, using default configuration")
        # Create minimal test configuration
        class DummyTestData:
            def __init__(self):
                self.word2id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "x_1": 3, "x_2": 4, "x_3": 5, 
                               "+": 6, "-": 7, "*": 8, "/": 9, "log": 10, "exp": 11, "sin": 12, "cos": 13, "tan": 14}
                self.id2word = {v: k for k, v in self.word2id.items()}
                self.una_ops = ["log", "exp", "sin", "cos", "tan"]
                self.bin_ops = ["+", "-", "*", "/"]
                self.total_variables = ["x_1", "x_2", "x_3"]
                self.total_coefficients = ["c_1", "c_2", "c_3"]
                self.rewrite_functions = []
        
        test_data = DummyTestData()
    
    # Load model
    model = Model.load_from_checkpoint(str(model_path), cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    # Setup BFGS parameters
    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    
    # Setup fit parameters
    params_fit = FitParams(
        word2id=test_data.word2id,
        id2word=test_data.id2word,
        una_ops=test_data.una_ops,
        bin_ops=test_data.bin_ops,
        total_variables=list(test_data.total_variables),
        total_coefficients=list(test_data.total_coefficients),
        rewrite_functions=list(test_data.rewrite_functions),
        bfgs=bfgs,
        beam_size=cfg.inference.beam_size
    )
    
    # Beam search configurations to test
    beam_configs = cfg.get('beam_configs', [
        {"beam_size": 1, "length_penalty": 1.0, "max_len": 50},
        {"beam_size": 3, "length_penalty": 1.0, "max_len": 75},
        {"beam_size": 5, "length_penalty": 1.0, "max_len": 100},
    ])
    
    # Load test equations
    test_csv_path = "test_set/nc.csv"
    print(f"Loading test equations from: {test_csv_path}")
    equations = load_test_equations_from_csv(test_csv_path)
    print(f"Loaded {len(equations)} test equations")
    
    # Run evaluation
    all_results = []
    max_equations = cfg.get('max_test_equations', 10)  # Limit for initial testing
    
    for i, eq in enumerate(equations[:max_equations]):
        print(f"\\n--- Equation {i+1}/{min(len(equations), max_equations)} ---")
        results = evaluate_equation(model, eq, params_fit, beam_configs)
        if results:
            all_results.extend(results)
    
    # Save results
    output_dir = Path(cfg.output_dir) / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "baseline_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\\nEvaluation Results Summary")
    print("=" * 50)
    print(f"Total equations tested: {len(set(r['equation_idx'] for r in all_results))}")
    print(f"Total beam configurations: {len(beam_configs)}")
    print(f"Results saved to: {results_file}")
    
    # Calculate summary statistics
    successful_predictions = [r for r in all_results if r.get('predicted') and r.get('mse') is not None]
    if successful_predictions:
        avg_mse = np.mean([r['mse'] for r in successful_predictions if np.isfinite(r['mse'])])
        avg_r2 = np.mean([r['r2'] for r in successful_predictions if np.isfinite(r['r2'])])
        symbolic_matches = sum(r.get('symbolic_match', False) for r in successful_predictions)
        
        print(f"Successful predictions: {len(successful_predictions)}/{len(all_results)}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average R²: {avg_r2:.6f}")
        print(f"Symbolic matches: {symbolic_matches}/{len(successful_predictions)}")
    
    print("\\nBaseline evaluation completed!")

if __name__ == "__main__":
    main()
