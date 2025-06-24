#!/usr/bin/env python3
"""
Unified evaluation script for NESYMRES models.
Integrates evaluation, benchmarking, and analysis capabilities.
"""

import os
import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import BFGSParams, FitParams, NNEquation
from nesymres import benchmark


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    r2_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    relative_error: float = 0.0
    exact_match_rate: float = 0.0
    runtime_seconds: float = 0.0


@dataclass
class BenchmarkResult:
    """Result from a single benchmark evaluation."""
    equation_id: str
    equation_expr: str
    predicted_expr: str
    metrics: EvaluationMetrics
    beam_size: int
    success: bool
    error_message: Optional[str] = None


class UnifiedEvaluator:
    """Unified evaluator with comprehensive benchmarking and analysis."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup output directory
        if hasattr(cfg, 'output_dir'):
            self.output_dir = Path(cfg.output_dir) / self.experiment_name
        else:
            self.output_dir = Path("evaluation_results") / self.experiment_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting evaluation: {self.experiment_name}")
        print(f"Output directory: {self.output_dir}")
        
        # Save configuration
        self._save_config()
        
        # Setup model and data
        self.model = self._load_model()
        self.data_module = self._setup_data()
        self.fit_params = self._setup_fit_params()
    
    def _save_config(self):
        """Save evaluation configuration."""
        config_path = self.output_dir / "eval_config.yaml"
        with open(config_path, 'w') as f:
            if hasattr(OmegaConf, 'save'):
                OmegaConf.save(self.cfg, f)
            else:
                import yaml
                yaml.dump(dict(self.cfg), f)
        
        config_json_path = self.output_dir / "eval_config.json"
        with open(config_json_path, 'w') as f:
            if hasattr(OmegaConf, 'to_container'):
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=2)
            else:
                json.dump(dict(self.cfg), f, indent=2)
    
    def _load_model(self) -> Model:
        """Load the model for evaluation."""
        model_path = Path(hydra.utils.to_absolute_path(self.cfg.model_path))
        print(f"Loading model from: {model_path}")
        
        model = Model.load_from_checkpoint(str(model_path), cfg=self.cfg.architecture)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        
        return model
    
    def _setup_data(self) -> DataModule:
        """Setup data module for evaluation."""
        test_path = Path(hydra.utils.to_absolute_path(self.cfg.test_path))
        
        if hasattr(self.cfg, 'data'):
            data_cfg = self.cfg.data
        else:
            data_cfg = self.cfg
        
        data_module = DataModule(None, None, test_path, data_cfg)
        data_module.setup()
        
        return data_module
    
    def _setup_fit_params(self) -> FitParams:
        """Setup fitting parameters."""
        test_data = load_metadata_hdf5(Path(hydra.utils.to_absolute_path(self.cfg.test_path)))
        
        bfgs = BFGSParams(
            activated=self.cfg.inference.bfgs.activated,
            n_restarts=self.cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=self.cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=self.cfg.inference.bfgs.normalization_o,
            idx_remove=self.cfg.inference.bfgs.idx_remove,
            normalization_type=self.cfg.inference.bfgs.normalization_type,
            stop_time=self.cfg.inference.bfgs.stop_time,
        )
        
        return FitParams(
            word2id=test_data.word2id,
            id2word=test_data.id2word,
            una_ops=test_data.una_ops,
            bin_ops=test_data.bin_ops,
            total_variables=list(test_data.total_variables),
            total_coefficients=list(test_data.total_coefficients),
            rewrite_functions=list(test_data.rewrite_functions),
            bfgs=bfgs,
            beam_size=self.cfg.inference.beam_size
        )
    
    def compute_metrics(self, true_values: np.ndarray, pred_values: np.ndarray, 
                       true_expr: str, pred_expr: str, runtime: float) -> EvaluationMetrics:
        """Compute comprehensive evaluation metrics."""
        try:
            # Numerical metrics
            r2 = r2_score(true_values, pred_values)
            mse = mean_squared_error(true_values, pred_values)
            mae = mean_absolute_error(true_values, pred_values)
            rmse = np.sqrt(mse)
            
            # Relative error
            rel_error = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-8)))
            
            # Symbolic accuracy (exact match)
            exact_match = 1.0 if str(true_expr).strip() == str(pred_expr).strip() else 0.0
            
            return EvaluationMetrics(
                r2_score=float(r2),
                mse=float(mse),
                mae=float(mae),
                rmse=float(rmse),
                relative_error=float(rel_error),
                exact_match_rate=exact_match,
                runtime_seconds=runtime
            )
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return EvaluationMetrics(runtime_seconds=runtime)
    
    def evaluate_single_equation(self, equation: NNEquation, beam_size: int) -> BenchmarkResult:
        """Evaluate a single equation with specified beam size."""
        try:
            X, y = equation.numerical_values[:-1], equation.numerical_values[-1:]
            
            if len(X.reshape(-1)) == 0:
                return BenchmarkResult(
                    equation_id=str(hash(str(equation.expr))),
                    equation_expr=str(equation.expr),
                    predicted_expr="SKIP_NO_DATA",
                    metrics=EvaluationMetrics(),
                    beam_size=beam_size,
                    success=False,
                    error_message="No valid data points"
                )
            
            # Update beam size
            self.fit_params.beam_size = beam_size
            fitfunc = partial(self.model.fitfunc, cfg_params=self.fit_params)
            
            # Run fitting with timing
            start_time = time.time()
            output = fitfunc(X.T, y.squeeze())
            runtime = time.time() - start_time
            
            # Extract prediction
            if isinstance(output, dict) and 'best_bfgs_preds' in output:
                pred_expr = output['best_bfgs_preds']
            else:
                pred_expr = str(output)
            
            # Compute metrics (simplified for now)
            metrics = EvaluationMetrics(
                exact_match_rate=1.0 if str(equation.expr).strip() == str(pred_expr).strip() else 0.0,
                runtime_seconds=runtime
            )
            
            return BenchmarkResult(
                equation_id=str(hash(str(equation.expr))),
                equation_expr=str(equation.expr),
                predicted_expr=str(pred_expr),
                metrics=metrics,
                beam_size=beam_size,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                equation_id=str(hash(str(equation.expr))),
                equation_expr=str(equation.expr),
                predicted_expr="ERROR",
                metrics=EvaluationMetrics(),
                beam_size=beam_size,
                success=False,
                error_message=str(e)
            )
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        print("Starting comprehensive evaluation...")
        
        # Get beam configurations
        if hasattr(self.cfg.inference, 'beam_configs'):
            beam_configs = self.cfg.inference.beam_configs
        else:
            beam_configs = [
                {"beam_size": 1, "length_penalty": 1.0, "max_len": 50},
                {"beam_size": 5, "length_penalty": 1.0, "max_len": 100},
                {"beam_size": 10, "length_penalty": 1.0, "max_len": 150},
            ]
        
        all_results = []
        equation_count = 0
        
        # Evaluate each equation with different beam sizes
        for batch in self.data_module.test_dataloader():
            if not len(batch[0]):
                continue
            
            equation = NNEquation(batch[0][0], batch[1][0], batch[2][0])
            equation_count += 1
            
            print(f"Evaluating equation {equation_count}: {equation.expr}")
            
            for config in beam_configs:
                result = self.evaluate_single_equation(equation, config["beam_size"])
                result_dict = {
                    "equation_count": equation_count,
                    "beam_config": config,
                    **result.__dict__
                }
                all_results.append(result_dict)
            
            # Save intermediate results
            if equation_count % 10 == 0:
                self._save_intermediate_results(all_results, equation_count)
        
        # Compile final results
        final_results = {
            "experiment_name": self.experiment_name,
            "total_equations": equation_count,
            "beam_configs": beam_configs,
            "results": all_results,
            "summary": self._compute_summary_stats(all_results)
        }
        
        # Save final results
        self._save_final_results(final_results)
        
        print(f"Evaluation completed!")
        print(f"Total equations: {equation_count}")
        print(f"Results saved to: {self.output_dir}")
        
        return final_results
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """Save intermediate results."""
        intermediate_path = self.output_dir / f"intermediate_results_{count}.json"
        with open(intermediate_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final evaluation results."""
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as CSV for easy analysis
        summary_path = self.output_dir / "evaluation_summary.csv"
        if "summary" in results:
            pd.DataFrame([results["summary"]]).to_csv(summary_path, index=False)
    
    def _compute_summary_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics."""
        if not results:
            return {}
        
        # Group by beam size
        beam_stats = {}
        for beam_config in set(r["beam_config"]["beam_size"] for r in results):
            beam_results = [r for r in results if r["beam_config"]["beam_size"] == beam_config]
            
            success_rate = sum(1 for r in beam_results if r["success"]) / len(beam_results)
            exact_match_rate = np.mean([r["metrics"]["exact_match_rate"] for r in beam_results])
            avg_runtime = np.mean([r["metrics"]["runtime_seconds"] for r in beam_results])
            
            beam_stats[f"beam_{beam_config}"] = {
                "success_rate": success_rate,
                "exact_match_rate": exact_match_rate,
                "avg_runtime": avg_runtime
            }
        
        return beam_stats


@hydra.main(config_name="config")
def main(cfg):
    """Main evaluation function."""
    evaluator = UnifiedEvaluator(cfg)
    results = evaluator.run_evaluation()
    
    print("\n=== Evaluation Summary ===")
    if "summary" in results:
        for beam_size, stats in results["summary"].items():
            print(f"{beam_size}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
