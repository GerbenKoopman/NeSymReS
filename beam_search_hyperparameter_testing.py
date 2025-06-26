#!/usr/bin/env python3
"""
Comprehensive Beam Search Hyperparameter Testing Script for NESYMRES

This script tests different beam search hyperparameters (beam_size, length_penalty, max_len)
against various metrics and outputs results as plots and CSV files.

Usage:
    python beam_search_hyperparameter_testing.py --config config_original.yaml --model_path weights/100M.ckpt
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import argparse
from pathlib import Path
from functools import partial
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import yaml
from sympy import lambdify, sympify
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, BFGSParams, Equation
from nesymres.benchmark import load_equation, get_robust_data, evaluate_func, return_order_variables
from sklearn.metrics import mean_squared_error


class BeamSearchTester:
    """Class for testing different beam search hyperparameters."""
    
    def __init__(self, config_path: str, model_path: str, output_dir: Optional[str] = None):
        """
        Initialize the beam search tester.
        
        Args:
            config_path: Path to configuration YAML file
            model_path: Path to trained model checkpoint
            output_dir: Output directory for results (default: auto-generated)
        """
        self.config_path = config_path
        self.model_path = model_path
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"beam_search_experiment_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Load configuration
        self.cfg = self._load_config()
        
        # Setup model and parameters
        self.model = self._load_model()
        self.fit_params = self._setup_fit_params()
        
        # Define hyperparameter ranges
        self.beam_sizes = [1, 3, 5, 10]
        self.max_lens = [30, 50, 75, 100]
        
        # Results storage
        self.results = []
        
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    def _load_model(self):
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")
        
        # Create a simple config object for model loading
        class Config:
            def __init__(self, cfg_dict):
                for key, value in cfg_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        arch_cfg = Config(self.cfg['architecture'])
        model = Model.load_from_checkpoint(self.model_path, cfg=arch_cfg)
        model.eval()
        
        if torch.cuda.is_available():
            model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
            
        return model
    
    def _setup_fit_params(self):
        """Setup fitting parameters from config."""
        # Setup BFGS parameters
        bfgs_cfg = self.cfg['inference']['bfgs']
        bfgs = BFGSParams(
            activated=bfgs_cfg['activated'],
            n_restarts=bfgs_cfg['n_restarts'],
            add_coefficients_if_not_existing=bfgs_cfg['add_coefficients_if_not_existing'],
            normalization_o=bfgs_cfg['normalization_o'],
            idx_remove=bfgs_cfg['idx_remove'],
            normalization_type=bfgs_cfg['normalization_type'],
            stop_time=int(float(bfgs_cfg['stop_time']))  # Handle scientific notation
        )
        
        # Try to load actual metadata, fallback to default if not available
        try:
            # Look for available metadata files
            metadata_path = None
            possible_paths = [
                'jupyter/100M/eq_setting.json',
                'jupyter/10MPaper/equation_config.json',
                'dataset_configuration.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    metadata_path = path
                    break
            
            if metadata_path and metadata_path.endswith('eq_setting.json'):
                # Load from eq_setting.json format
                with open(metadata_path, 'r') as f:
                    eq_setting = json.load(f)
                
                word2id = eq_setting['word2id']
                id2word = {int(k): v for k, v in eq_setting['id2word'].items()}
                una_ops = eq_setting.get('una_ops', ['sin', 'cos', 'exp', 'ln', 'sqrt', 'abs', 'tan'])
                bin_ops = eq_setting.get('bin_ops', ['add', 'mul', 'sub', 'div', 'pow'])
                total_variables = eq_setting['total_variables']
                total_coefficients = eq_setting['total_coefficients']
                rewrite_functions = eq_setting.get('rewrite_functions', [])
                
                print(f"Loaded metadata from: {metadata_path}")
                
            else:
                raise FileNotFoundError("No suitable metadata file found")
                
        except Exception as e:
            print(f"Warning: Could not load metadata ({e}), using defaults")
            # Fallback to basic vocabulary
            word2id = {
                'P': 0, 'S': 1, 'F': 2, 'c': 3, 'x_1': 4, 'x_2': 5, 'x_3': 6,
                'add': 9, 'mul': 18, 'sub': 25, 'div': 15, 'pow': 19,
                'sin': 20, 'cos': 12, 'exp': 16, 'ln': 17, 'sqrt': 22, 'abs': 7,
                'tan': 23, '1': 29, '2': 30, '3': 31, '-1': 27, '-2': 26, '-3': 25
            }
            id2word = {v: k for k, v in word2id.items()}
            una_ops = ['sin', 'cos', 'exp', 'ln', 'sqrt', 'abs', 'tan']
            bin_ops = ['add', 'mul', 'sub', 'div', 'pow']
            total_variables = ['x_1', 'x_2', 'x_3']
            total_coefficients = ['c']
            rewrite_functions = []
        
        return FitParams(
            word2id=word2id,
            id2word=id2word,
            una_ops=una_ops,
            bin_ops=bin_ops,
            total_variables=total_variables,
            total_coefficients=total_coefficients,
            rewrite_functions=rewrite_functions,
            bfgs=bfgs,
            beam_size=5  # Default, will be overridden
        )
    
    def load_equations_from_csv(self, csv_path: str, max_equations: Optional[int] = None) -> List[Dict]:
        """Load equations from CSV file."""
        print(f"Loading equations from {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        equations = []
        for idx, row in df.iterrows():
            if max_equations and len(equations) >= max_equations:
                break
                
            try:
                # Parse the equation
                expr = row['eq']
                
                # Parse support (it's stored as a string representation of a dict)
                support_str = row['support']
                support = eval(support_str)  # Safe since we control the CSV
                
                # Extract variables from support
                variables = set(support.keys())
                
                # Get number of data points
                num_points = int(row['num_points']) if 'num_points' in row else 500
                
                equation_dict = {
                    'expr': expr,
                    'variables': variables,
                    'support': support,
                    'num_points': num_points,
                    'name': f'eq_{idx}',
                    'csv_index': idx
                }
                
                equations.append(equation_dict)
                
            except Exception as e:
                print(f"Error processing equation {idx}: {e}")
                continue
        
        print(f"Successfully loaded {len(equations)} equations from CSV")
        return equations
    
    def generate_data_for_equation(self, eq_dict: Dict, num_points: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data points for an equation."""
        variables = list(eq_dict['variables'])
        support = eq_dict['support']
        
        # Use num_points from equation dict if not specified
        if num_points is None:
            num_points = eq_dict.get('num_points', 100)
        
        # Ensure num_points is an integer
        num_points = int(num_points) if num_points is not None else 100
        
        # Generate random points within support, avoiding problematic values
        X_data = []
        for var in self.fit_params.total_variables:
            if var in support:
                var_min = support[var]['min']
                var_max = support[var]['max']
                
                # Generate points with some safety margin for divisions and logs
                if var_min < 0 and var_max > 0:
                    # Avoid values too close to zero for divisions
                    points = []
                    while len(points) < num_points:
                        candidates = np.random.uniform(var_min, var_max, num_points * 2)
                        # Filter out values too close to problematic points
                        safe_candidates = candidates[np.abs(candidates) > 0.01]
                        points.extend(safe_candidates[:num_points - len(points)])
                    points = np.array(points[:num_points])
                else:
                    points = np.random.uniform(var_min, var_max, num_points)
            else:
                points = np.zeros(num_points)  # Unused variables set to 0
            X_data.append(points)
        
        X = np.stack(X_data, axis=1)  # shape: (num_points, num_variables)
        
        # Evaluate equation with better error handling
        try:
            y = self.safe_evaluate_expression(eq_dict['expr'], list(variables), X)
        except Exception as e:
            print(f"Error evaluating equation {eq_dict['expr']}: {e}")
            y = np.random.randn(num_points)  # Fallback
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(np.array(y)).float()
        
        return X_tensor, y_tensor
    
    def evaluate_beam_config(self, beam_size: int, max_len: int, 
                           X: torch.Tensor, y: torch.Tensor, true_expr: str) -> Dict:
        """Evaluate a single beam search configuration."""
        # Update parameters
        self.fit_params.beam_size = beam_size
        self.fit_params.max_len = max_len
        
        # Setup beam config for model
        beam_configs = [{
            "beam_size": beam_size,
            "max_len": max_len
        }]
        
        try:
            # Run inference
            start_time = time.time()
            results = self.model.fitfunc(X, y.squeeze(), cfg_params=self.fit_params, beam_configs=beam_configs)
            inference_time = time.time() - start_time
            
            if results and len(results) > 0:
                result = results[0]  # Take first (and only) result
                output = result['output']
                
                if output['best_bfgs_preds'] and len(output['best_bfgs_preds']) > 0:
                    predicted_expr = str(output['best_bfgs_preds'][0])
                    best_loss = output['best_bfgs_loss'][0] if output['best_bfgs_loss'] else float('inf')
                    
                    # Try to evaluate predicted expression for metrics
                    try:
                        # Handle NESYMRES variable format (x_1, x_2, etc.)
                        pred_expr_clean = predicted_expr
                        for old_var, new_var in zip(['x_1', 'x_2', 'x_3'], self.fit_params.total_variables):
                            pred_expr_clean = pred_expr_clean.replace(old_var, new_var)
                        
                        pred_func = lambdify(self.fit_params.total_variables, sympify(pred_expr_clean), "numpy")
                        y_pred = pred_func(*X.T)
                        
                        if np.isscalar(y_pred):
                            y_pred = np.full(len(y), y_pred)
                        
                        # Convert to proper numpy arrays
                        y_true = y.numpy().flatten()
                        y_pred = np.array(y_pred).flatten()
                        
                        # Check for valid predictions
                        if len(y_pred) != len(y_true):
                            raise ValueError(f"Prediction length mismatch: {len(y_pred)} vs {len(y_true)}")
                        
                        # Filter out invalid values
                        valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred) | np.isnan(y_true) | np.isinf(y_true))
                        
                        if np.sum(valid_mask) < len(y_true) * 0.5:  # Less than 50% valid predictions
                            print(f"Warning: Too many invalid predictions, using fallback metrics")
                            mse = float('inf')
                        else:
                            y_true_valid = y_true[valid_mask]
                            y_pred_valid = y_pred[valid_mask]
                            try:
                                mse = mean_squared_error(y_true_valid, y_pred_valid)
                                # Clip MSE to reasonable range
                                mse = np.clip(mse, 0.0, 1e10)
                            except:
                                mse = float('inf')
                        
                    except Exception as e:
                        print(f"Error evaluating predicted expression: {e}")
                        y_pred = np.zeros_like(y.numpy())
                        mse = float('inf')
                
                else:
                    predicted_expr = "FAILED"
                    best_loss = float('inf')
                    mse = float('inf')
            else:
                predicted_expr = "NO_RESULT"
                best_loss = float('inf')
                mse = float('inf')
                
        except Exception as e:
            print(f"Error during inference: {e}")
            predicted_expr = "ERROR"
            best_loss = float('inf')
            mse = float('inf')
            inference_time = 0.0
        
        return {
            'beam_size': beam_size,
            'max_len': max_len,
            'predicted_expr': predicted_expr,
            'bfgs_loss': best_loss,
            'mse': mse,
            'inference_time': inference_time
        }
    
    def run_hyperparameter_sweep(self, max_configs: int = 50, num_equations: int = 5, csv_path: str = "test_set/nc.csv"):
        """Run comprehensive hyperparameter sweep."""
        print("Starting hyperparameter sweep...")
        print(f"Testing {len(self.beam_sizes)} beam sizes, {len(self.max_lens)} max lengths")
        
        # Load test equations from CSV
        test_equations = self.load_equations_from_csv(csv_path, num_equations)
        print(f"Loaded {len(test_equations)} test equations from CSV")
        
        # Create all combinations and sample if too many
        all_configs = list(product(self.beam_sizes, self.max_lens))
        if len(all_configs) > max_configs:
            print(f"Sampling {max_configs} configurations from {len(all_configs)} total combinations")
            indices = np.random.choice(len(all_configs), max_configs, replace=False)
            configs_to_test = [all_configs[i] for i in indices]
        else:
            configs_to_test = all_configs
        
        print(f"Testing {len(configs_to_test)} configurations")
        
        total_experiments = len(configs_to_test) * len(test_equations)
        experiment_count = 0
        
        for eq_idx, eq_dict in enumerate(test_equations):
            print(f"\nTesting equation {eq_idx + 1}/{len(test_equations)}: {eq_dict['name']}")
            print(f"  Expression: {eq_dict['expr']}")
            
            # Generate data for this equation
            try:
                X, y = self.generate_data_for_equation(eq_dict)
                print(f"  Generated {len(X)} data points")
            except Exception as e:
                print(f"  Error generating data: {e}, skipping equation")
                continue
            
            for config_idx, (beam_size, max_len) in enumerate(configs_to_test):
                experiment_count += 1
                print(f"  Config {config_idx + 1}/{len(configs_to_test)} (Overall: {experiment_count}/{total_experiments}): "
                      f"beam_size={beam_size}, max_len={max_len}")
                
                # Evaluate configuration
                result = self.evaluate_beam_config(beam_size, max_len, X, y, eq_dict['expr'])
                
                # Add equation info to result
                result.update({
                    'equation_idx': eq_idx,
                    'equation_name': eq_dict['name'],
                    'true_expr': eq_dict['expr'],
                    'csv_index': eq_dict.get('csv_index', -1),
                    'num_variables': len(eq_dict['variables'])
                })
                
                self.results.append(result)
                
                # Save intermediate results
                if experiment_count % 10 == 0:
                    self.save_results()
        
        print(f"\nCompleted {total_experiments} experiments")
        self.save_results()
    
    def save_results(self):
        """Save results to CSV and JSON files."""
        if not self.results:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = self.output_dir / "beam_search_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save as JSON
        json_path = self.output_dir / "beam_search_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {json_path}")
    
    def create_plots(self):
        """Create comprehensive plots of the results."""
        if not self.results:
            print("No results to plot")
            return
            
        df = pd.DataFrame(self.results)
        
        # Check if we have any valid results
        valid_mse = df[df['mse'] != float('inf')]
        if len(valid_mse) == 0:
            print("Warning: No valid results found. All inference attempts failed.")
            print("Creating diagnostic plots only...")
            
            # Create images directory in main repo
            images_dir = Path("/home/gerben-koopman/studie/nesymres/images")
            images_dir.mkdir(exist_ok=True)
            
            # Create a simple diagnostic plot showing the configuration attempts
            plt.figure(figsize=(12, 8))
            config_counts = df.groupby(['beam_size', 'max_len']).size().reset_index(name='count')
            plt.scatter(config_counts['beam_size'], config_counts['max_len'], 
                       s=config_counts['count']*50, alpha=0.6)
            plt.xlabel('Beam Size')
            plt.ylabel('Max Length')
            plt.title('Configuration Attempts (size = number of attempts)')
            plt.grid(True, alpha=0.3)
            plt.savefig(images_dir / "configuration_attempts.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Error rate plot
            plt.figure(figsize=(10, 6))
            error_types = df['predicted_expr'].value_counts()
            plt.pie(error_types.values, labels=[str(label) for label in error_types.index], autopct='%1.1f%%')
            plt.title('Distribution of Error Types')
            plt.savefig(images_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Diagnostic plots saved to /images directory")
            return
        
        # Setup plotting
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create images directory in main repo
        images_dir = Path("/home/gerben-koopman/studie/nesymres/images")
        images_dir.mkdir(exist_ok=True)
        
        # Define metrics to plot
        metrics = ['mse', 'bfgs_loss', 'inference_time']
        metric_labels = {
            'mse': 'MSE',
            'bfgs_loss': 'BFGS Loss',
            'inference_time': 'Inference Time (s)'
        }
        
        # Create heatmaps for each metric
        print("Creating heatmaps...")
        for metric in metrics:
            # Filter out invalid values for this metric
            if metric in ['mse', 'bfgs_loss']:
                valid_data = df[df[metric] != float('inf')]
            else:
                valid_data = df.copy()
            
            if len(valid_data) == 0:
                print(f"No valid data for {metric}, skipping heatmap")
                continue
            
            plt.figure(figsize=(12, 8))
            try:
                # Create pivot table with beam_size as columns and max_len as rows
                pivot_data = valid_data.pivot_table(values=metric, index='max_len', columns='beam_size', aggfunc='mean')
                
                if not pivot_data.empty and pivot_data.shape[0] > 0 and pivot_data.shape[1] > 0:
                    # Choose colormap based on metric (lower is better for MSE and BFGS loss)
                    if metric in ['mse', 'bfgs_loss']:
                        cmap = 'viridis_r'  # Reverse colormap so darker = better (lower)
                    else:
                        cmap = 'viridis'   # Regular colormap so darker = better (higher)
                    
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, cbar_kws={'label': metric_labels[metric]})
                    plt.title(f'{metric_labels[metric]} Heatmap: Beam Size vs Max Length')
                    plt.ylabel('Max Length')
                    plt.xlabel('Beam Size')
                    plt.tight_layout()
                    plt.savefig(images_dir / f"heatmap_{metric}.png", dpi=300, bbox_inches='tight')
                else:
                    plt.text(0.5, 0.5, f'No valid {metric_labels[metric]} data to plot', 
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                    plt.title(f'{metric_labels[metric]} Heatmap (No Data)')
                    plt.savefig(images_dir / f"heatmap_{metric}.png", dpi=300, bbox_inches='tight')
            except Exception as e:
                plt.text(0.5, 0.5, f'Error creating {metric} heatmap: {str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.title(f'{metric_labels[metric]} Heatmap (Error)')
                plt.savefig(images_dir / f"heatmap_{metric}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create line plots for beam size vs metrics
        print("Creating line plots for beam size...")
        for metric in metrics:
            # Filter out invalid values
            if metric in ['mse', 'bfgs_loss']:
                valid_data = df[df[metric] != float('inf')]
            else:
                valid_data = df.copy()
            
            if len(valid_data) == 0:
                print(f"No valid data for {metric}, skipping beam size line plot")
                continue
            
            beam_stats = valid_data.groupby('beam_size')[metric].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot without confidence intervals
            plt.figure(figsize=(10, 6))
            plt.plot(beam_stats['beam_size'], beam_stats['mean'], marker='o', linewidth=2, markersize=8)
            plt.xlabel('Beam Size')
            plt.ylabel(metric_labels[metric])
            plt.title(f'{metric_labels[metric]} vs Beam Size')
            plt.grid(True, alpha=0.3)
            if metric in ['mse', 'bfgs_loss']:
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(images_dir / f"line_{metric}_vs_beam_size.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot with confidence intervals
            plt.figure(figsize=(10, 6))
            plt.errorbar(beam_stats['beam_size'], beam_stats['mean'], yerr=beam_stats['std'], 
                        marker='o', capsize=5, linewidth=2, markersize=8)
            plt.xlabel('Beam Size')
            plt.ylabel(metric_labels[metric])
            plt.title(f'{metric_labels[metric]} vs Beam Size (with confidence intervals)')
            plt.grid(True, alpha=0.3)
            if metric in ['mse', 'bfgs_loss']:
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(images_dir / f"line_{metric}_vs_beam_size_ci.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create line plots for max length vs metrics
        print("Creating line plots for max length...")
        for metric in metrics:
            # Filter out invalid values
            if metric in ['mse', 'bfgs_loss']:
                valid_data = df[df[metric] != float('inf')]
            else:
                valid_data = df.copy()
            
            if len(valid_data) == 0:
                print(f"No valid data for {metric}, skipping max length line plot")
                continue
            
            maxlen_stats = valid_data.groupby('max_len')[metric].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot without confidence intervals
            plt.figure(figsize=(10, 6))
            plt.plot(maxlen_stats['max_len'], maxlen_stats['mean'], marker='s', linewidth=2, markersize=8)
            plt.xlabel('Max Length')
            plt.ylabel(metric_labels[metric])
            plt.title(f'{metric_labels[metric]} vs Max Length')
            plt.grid(True, alpha=0.3)
            if metric in ['mse', 'bfgs_loss']:
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(images_dir / f"line_{metric}_vs_max_len.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot with confidence intervals
            plt.figure(figsize=(10, 6))
            plt.errorbar(maxlen_stats['max_len'], maxlen_stats['mean'], yerr=maxlen_stats['std'], 
                        marker='s', capsize=5, linewidth=2, markersize=8)
            plt.xlabel('Max Length')
            plt.ylabel(metric_labels[metric])
            plt.title(f'{metric_labels[metric]} vs Max Length (with confidence intervals)')
            plt.grid(True, alpha=0.3)
            if metric in ['mse', 'bfgs_loss']:
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(images_dir / f"line_{metric}_vs_max_len_ci.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create combined comparison plots
        print("Creating combined comparison plots...")
        
        # Combined line plot for all metrics vs beam size (without CI)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in ['mse', 'bfgs_loss']:
                valid_data = df[df[metric] != float('inf')]
            else:
                valid_data = df.copy()
            
            if len(valid_data) > 0:
                beam_stats = valid_data.groupby('beam_size')[metric].agg(['mean']).reset_index()
                axes[i].plot(beam_stats['beam_size'], beam_stats['mean'], marker='o', linewidth=2)
                axes[i].set_xlabel('Beam Size')
                axes[i].set_ylabel(metric_labels[metric])
                axes[i].set_title(f'{metric_labels[metric]} vs Beam Size')
                axes[i].grid(True, alpha=0.3)
                if metric in ['mse', 'bfgs_loss']:
                    axes[i].set_yscale('log')
            else:
                axes[i].text(0.5, 0.5, f'No valid {metric} data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{metric_labels[metric]} vs Beam Size (No Data)')
        
        plt.tight_layout()
        plt.savefig(images_dir / "combined_metrics_vs_beam_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined line plot for all metrics vs max length (without CI)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in ['mse', 'bfgs_loss']:
                valid_data = df[df[metric] != float('inf')]
            else:
                valid_data = df.copy()
            
            if len(valid_data) > 0:
                maxlen_stats = valid_data.groupby('max_len')[metric].agg(['mean']).reset_index()
                axes[i].plot(maxlen_stats['max_len'], maxlen_stats['mean'], marker='s', linewidth=2)
                axes[i].set_xlabel('Max Length')
                axes[i].set_ylabel(metric_labels[metric])
                axes[i].set_title(f'{metric_labels[metric]} vs Max Length')
                axes[i].grid(True, alpha=0.3)
                if metric in ['mse', 'bfgs_loss']:
                    axes[i].set_yscale('log')
            else:
                axes[i].text(0.5, 0.5, f'No valid {metric} data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{metric_labels[metric]} vs Max Length (No Data)')
        
        plt.tight_layout()
        plt.savefig(images_dir / "combined_metrics_vs_max_len.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution plots
        print("Creating distribution plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in ['mse', 'bfgs_loss']:
                valid_data = df[df[metric] != float('inf')]
            else:
                valid_data = df.copy()
            
            if len(valid_data) > 0:
                axes[i].hist(valid_data[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(metric_labels[metric])
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {metric_labels[metric]}')
                if metric in ['mse', 'bfgs_loss']:
                    axes[i].set_xscale('log')
            else:
                axes[i].text(0.5, 0.5, f'No valid {metric} data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Distribution of {metric_labels[metric]} (No Data)')
        
        plt.tight_layout()
        plt.savefig(images_dir / "metric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"All plots saved to {images_dir}")
        print(f"Generated plots:")
        print(f"  - 4 heatmaps: heatmap_{{metric}}.png")
        print(f"  - 8 line plots without CI: line_{{metric}}_vs_{{param}}.png")
        print(f"  - 8 line plots with CI: line_{{metric}}_vs_{{param}}_ci.png")
        print(f"  - 2 combined plots: combined_metrics_vs_{{param}}.png")
        print(f"  - 1 distribution plot: metric_distributions.png")
    
    def create_summary_report(self):
        """Create a summary report of the results."""
        if not self.results:
            print("No results to summarize")
            return
            
        df = pd.DataFrame(self.results)
        
        # Best configurations for each metric
        summary = {
            'total_experiments': len(df),
            'unique_equations': df['equation_name'].nunique(),
            'best_configs': {}
        }
        
        # Helper function to safely extract values
        def safe_extract(row, column):
            try:
                val = row[column]
                if pd.isna(val):
                    return 0.0
                return float(val)
            except (TypeError, ValueError):
                return 0.0
        
        # Best (lowest) MSE
        try:
            best_mse_idx = df['mse'].idxmin()
            if not pd.isna(best_mse_idx) and best_mse_idx in df.index:
                best_mse_row = df.loc[best_mse_idx]
                summary['best_configs']['mse'] = {
                    'beam_size': safe_extract(best_mse_row, 'beam_size'),
                    'length_penalty': safe_extract(best_mse_row, 'length_penalty'),
                    'max_len': safe_extract(best_mse_row, 'max_len'),
                    'value': safe_extract(best_mse_row, 'mse'),
                    'equation': str(best_mse_row['equation_name'])
                }
        except Exception as e:
            print(f"Error processing best MSE config: {e}")
        
        # Highest symbolic match rate
        try:
            best_symbolic_idx = df['symbolic_match'].idxmax()
            if not pd.isna(best_symbolic_idx) and best_symbolic_idx in df.index:
                best_symbolic_row = df.loc[best_symbolic_idx]
                summary['best_configs']['symbolic_match'] = {
                    'beam_size': safe_extract(best_symbolic_row, 'beam_size'),
                    'length_penalty': safe_extract(best_symbolic_row, 'length_penalty'),
                    'max_len': safe_extract(best_symbolic_row, 'max_len'),
                    'value': safe_extract(best_symbolic_row, 'symbolic_match'),
                    'equation': str(best_symbolic_row['equation_name'])
                }
        except Exception as e:
            print(f"Error processing best symbolic config: {e}")
        
        # Average metrics by hyperparameter
        try:
            summary['averages'] = {
                'by_beam_size': df.groupby('beam_size')[['mse', 'inference_time']].mean().to_dict(),
                'by_max_len': df.groupby('max_len')[['mse', 'inference_time']].mean().to_dict()
            }
        except Exception as e:
            print(f"Error computing averages: {e}")
            summary['averages'] = {}
        
        # General statistics
        try:
            summary['statistics'] = {
                'mse': {'mean': float(df['mse'].mean()), 'std': float(df['mse'].std())},
                'average_inference_time': float(df['inference_time'].mean())
            }
        except Exception as e:
            print(f"Error computing statistics: {e}")
            summary['statistics'] = {}
        
        # Save summary
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create readable text summary
        text_summary_path = self.output_dir / "experiment_summary.txt"
        with open(text_summary_path, 'w') as f:
            f.write("Beam Search Hyperparameter Testing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Experiments: {summary['total_experiments']}\n")
            f.write(f"Unique Equations Tested: {summary['unique_equations']}\n\n")
            
            f.write("Best Configurations:\n")
            f.write("-" * 20 + "\n")
            
            if 'mse' in summary['best_configs']:
                config = summary['best_configs']['mse']
                f.write(f"Best (lowest) MSE: {config['value']:.6f}\n")
                f.write(f"  - Beam Size: {config['beam_size']}\n")
                f.write(f"  - Length Penalty: {config['length_penalty']}\n")
                f.write(f"  - Max Length: {config['max_len']}\n")
                f.write(f"  - Equation: {config['equation']}\n\n")
            
            if 'symbolic_match' in summary['best_configs']:
                config = summary['best_configs']['symbolic_match']
                f.write(f"Best Symbolic Match: {config['value']:.2f}\n")
                f.write(f"  - Beam Size: {config['beam_size']}\n")
                f.write(f"  - Length Penalty: {config['length_penalty']}\n")
                f.write(f"  - Max Length: {config['max_len']}\n")
                f.write(f"  - Equation: {config['equation']}\n\n")
            
            if summary['statistics']:
                f.write("Overall Statistics:\n")
                f.write("-" * 17 + "\n")
                stats = summary['statistics']
                f.write(f"Average MSE: {stats.get('mse', {}).get('mean', 0):.6f} ± {stats.get('mse', {}).get('std', 0):.6f}\n")
                f.write(f"Average MAE: {stats.get('mae', {}).get('mean', 0):.6f} ± {stats.get('mae', {}).get('std', 0):.6f}\n")
                f.write(f"Symbolic Match Rate: {stats.get('symbolic_match_rate', 0):.4f}\n")
                f.write(f"Average Inference Time: {stats.get('average_inference_time', 0):.4f} seconds\n")
        
        print(f"Summary saved to {summary_path} and {text_summary_path}")
    
    def safe_evaluate_expression(self, expr_str: str, variables: List[str], X: np.ndarray, max_retries: int = 3) -> np.ndarray:
        """Safely evaluate an expression with multiple retry strategies."""
        for attempt in range(max_retries):
            try:
                expr = sympify(expr_str)
                
                # Create variable mapping for evaluation
                var_mapping = {}
                for var in variables:
                    if var in self.fit_params.total_variables:
                        var_idx = self.fit_params.total_variables.index(var)
                        var_mapping[var] = X[:, var_idx]
                    else:
                        var_mapping[var] = np.zeros(X.shape[0])
                
                # Use lambdify for safe evaluation
                func = lambdify(list(variables), expr, "numpy")
                
                if len(variables) == 1:
                    y = func(var_mapping[variables[0]])
                else:
                    y = func(*[var_mapping[var] for var in variables])
                
                # Handle scalar results
                if np.isscalar(y):
                    y = np.full(X.shape[0], y)
                
                # Check for invalid values and replace them
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    print(f"Warning: Invalid values found in attempt {attempt + 1}, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # Replace invalid values with finite ones
                        y = np.where(np.isnan(y) | np.isinf(y), np.random.randn(len(y)), y)
                
                return y
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    break
        
        # Fallback
        print(f"All attempts failed for expression: {expr_str}")
        return np.random.randn(X.shape[0])


def main():
    """Main function to run the beam search hyperparameter testing."""
    parser = argparse.ArgumentParser(description="Test beam search hyperparameters for NESYMRES")
    parser.add_argument("--config", default="config_original.yaml", help="Path to config file")
    parser.add_argument("--model_path", default="weights/100M.ckpt", help="Path to model checkpoint")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--max_configs", type=int, default=50, help="Maximum number of configurations to test")
    parser.add_argument("--num_equations", type=int, default=5, help="Number of test equations")
    parser.add_argument("--csv_path", default="test_set/nc.csv", help="Path to CSV file with equations")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file {args.csv_path} not found")
        return
    
    # Initialize tester
    print("Initializing Beam Search Hyperparameter Tester...")
    tester = BeamSearchTester(args.config, args.model_path, args.output_dir)
    
    # Run experiments
    print("Starting hyperparameter sweep...")
    tester.run_hyperparameter_sweep(max_configs=args.max_configs, num_equations=args.num_equations, csv_path=args.csv_path)
    
    # Create plots and summary
    print("Creating plots...")
    tester.create_plots()
    
    print("Creating summary report...")
    tester.create_summary_report()
    
    print(f"Beam search hyperparameter testing completed!")
    print(f"Results saved in: {tester.output_dir}")


if __name__ == "__main__":
    main()
