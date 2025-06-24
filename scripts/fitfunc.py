import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from nesymres.architectures.bfgs import bfgs
from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.dclasses import BFGSParams, FitParams, NNEquation
from nesymres.utils import load_metadata_hdf5
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf

def setup_experiment_dir(cfg):
    """Setup experiment directory for fitfunc evaluation."""
    experiment_name = f"fitfunc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if hasattr(cfg, 'output_dir'):
        output_dir = Path(cfg.output_dir) / experiment_name
    else:
        output_dir = Path("evaluation_results") / experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return experiment_name, output_dir

def save_config(cfg, output_dir):
    """Save experiment configuration."""
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        if hasattr(OmegaConf, 'save'):
            OmegaConf.save(cfg, f)
        else:
            import yaml
            yaml.dump(dict(cfg), f)
            
    config_json_path = output_dir / "config.json"
    with open(config_json_path, 'w') as f:
        if hasattr(OmegaConf, 'to_container'):
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
        else:
            json.dump(dict(cfg), f, indent=2)

def save_results(results, output_dir, equation_idx):
    """Save evaluation results."""
    results_path = output_dir / f"results_eq_{equation_idx}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

@hydra.main(config_name="config")
def main(cfg):
    # Setup experiment
    experiment_name, output_dir = setup_experiment_dir(cfg)
    print(f"Starting fitfunc evaluation: {experiment_name}")
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    save_config(cfg, output_dir)
    
    # Set random seed
    random_seed = getattr(cfg, 'random_seed', 42)
    seed_everything(random_seed, workers=True)
    
    # Load model and test data
    model_path = Path(hydra.utils.to_absolute_path(cfg.model_path))
    test_data = load_metadata_hdf5(Path(hydra.utils.to_absolute_path(cfg.test_path)))

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

    # Setup data module
    data = DataModule(
        None,
        None,
        Path(hydra.utils.to_absolute_path(cfg.test_path)),
        cfg
    )

    data.setup()
    
    # Load model
    model = Model.load_from_checkpoint(model_path, cfg=cfg.architecture)
    model.eval()
    model.cuda()
    
    # Setup beam search configurations for comprehensive evaluation
    beam_configs = getattr(cfg.inference, 'beam_configs', [
        {"beam_size": 1, "length_penalty": 1.0, "max_len": 50},
        {"beam_size": 3, "length_penalty": 1.0, "max_len": 75},
        {"beam_size": 5, "length_penalty": 1.0, "max_len": 100},
        {"beam_size": 10, "length_penalty": 1.0, "max_len": 150},
    ])
    
    # Process equations
    all_results = []
    equation_idx = 0
    
    for batch in data.test_dataloader():
        if not len(batch[0]):
            continue
            
        eq = NNEquation(batch[0][0], batch[1][0], batch[2][0])
        X, y = eq.numerical_values[:-1], eq.numerical_values[-1:]
        
        if len(X.reshape(-1)) == 0:
            print(f"Skipping equation {equation_idx} because no points are valid")
            continue
            
        print(f"Testing equation {equation_idx}: {eq.expr}")

        # Evaluate with different beam search configurations
        equation_results = {
            "equation_idx": equation_idx,
            "expression": str(eq.expr),
            "data_shape": {"X": X.shape, "y": y.shape},
            "beam_results": []
        }
        
        for config_idx, config in enumerate(beam_configs):
            print(f"  Config {config_idx + 1}/{len(beam_configs)}: {config}")
            
            # Update beam size for this configuration
            params_fit.beam_size = config["beam_size"]
            fitfunc = partial(model.fitfunc, cfg_params=params_fit)
            
            try:
                # Run fitting
                start_time = time.time()
                output = fitfunc(X.T, y.squeeze())
                elapsed_time = time.time() - start_time
                
                # Store results
                beam_result = {
                    "config": config,
                    "elapsed_time": elapsed_time,
                    "output": output,
                    "best_prediction": output.get('best_bfgs_preds', 'N/A') if isinstance(output, dict) else str(output)
                }
                equation_results["beam_results"].append(beam_result)
                
                print(f"    Prediction: {beam_result['best_prediction']}")
                print(f"    Time: {elapsed_time:.2f}s")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                beam_result = {
                    "config": config,
                    "error": str(e),
                    "elapsed_time": None,
                    "output": None
                }
                equation_results["beam_results"].append(beam_result)
        
        # Save results for this equation
        save_results(equation_results, output_dir, equation_idx)
        all_results.append(equation_results)
        equation_idx += 1
        
        print(f"Completed equation {equation_idx}/{equation_idx} evaluations")
    
    # Save comprehensive results
    final_results = {
        "experiment_name": experiment_name,
        "total_equations": len(all_results),
        "beam_configs": beam_configs,
        "results": all_results
    }
    
    final_results_path = output_dir / "complete_results.json"
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Total equations processed: {len(all_results)}")
    print(f"Beam configurations tested: {len(beam_configs)}")
        
if __name__ == "__main__":
    main()