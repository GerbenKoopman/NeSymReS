import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import BFGSParams, FitParams
import hydra


def setup_loggers(cfg, experiment_name, output_dir):
    """Setup multiple loggers for comprehensive monitoring."""
    loggers = []
    
    # CSV Logger (always enabled)
    csv_logger = CSVLogger(
        save_dir=str(output_dir),
        name="csv_logs"
    )
    loggers.append(csv_logger)
    
    # TensorBoard Logger 
    if hasattr(cfg, 'logging') and getattr(cfg.logging, 'tensorboard', False):
        tb_logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name="tb_logs"
        )
        loggers.append(tb_logger)
    
    # Weights & Biases Logger
    if cfg.wandb:
        import wandb
        wandb.init(project="nesymres")
        wandb_logger = WandbLogger(
            project=getattr(cfg, 'wandb_project', 'nesymres'),
            name=experiment_name,
            save_dir=str(output_dir)
        )
        loggers.append(wandb_logger)
        
    return loggers

def setup_callbacks(cfg, output_dir):
    """Setup training callbacks."""
    callbacks = []
    
    # Model Checkpoint
    enable_checkpointing = getattr(cfg, 'enable_checkpointing', True)
    if enable_checkpointing:
        save_top_k = getattr(cfg, 'save_top_k', 3)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints") if hasattr(cfg, 'output_dir') else "Exp_weights/",
            filename="{epoch:02d}-{val_loss:.4f}" if hasattr(cfg, 'output_dir') else None,
            monitor="val_loss",
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
    
    # Early Stopping
    early_stopping_patience = getattr(cfg, 'early_stopping_patience', 0)
    if early_stopping_patience > 0:
        early_stopping_min_delta = getattr(cfg, 'early_stopping_min_delta', 0.001)
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks

def save_config(cfg, output_dir):
    """Save experiment configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        if hasattr(OmegaConf, 'save'):
            OmegaConf.save(cfg, f)
        else:
            # Fallback for older versions
            import yaml
            yaml.dump(dict(cfg), f)
            
    # Also save as JSON for easier parsing
    config_json_path = output_dir / "config.json"
    with open(config_json_path, 'w') as f:
        if hasattr(OmegaConf, 'to_container'):
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
        else:
            json.dump(dict(cfg), f, indent=2)


def setup_model(cfg, finetune_mode=False):
    """Setup model with optional fine-tuning from checkpoint."""
    if finetune_mode and hasattr(cfg, 'finetune') and cfg.finetune.enabled:
        print("Setting up model for fine-tuning...")
        
        # Load base model from checkpoint
        base_model_path = Path(hydra.utils.to_absolute_path(cfg.finetune.base_model_path))
        print(f"Loading base model from: {base_model_path}")
        
        model = Model.load_from_checkpoint(
            str(base_model_path),
            cfg=cfg.architecture
        )
        
        # Apply fine-tuning strategy
        if cfg.finetune.freeze_encoder:
            print("Freezing encoder layers...")
            for param in model.encoder.parameters():
                param.requires_grad = False
        
        # Freeze specific layers if specified
        if hasattr(cfg.finetune, 'freeze_layers') and cfg.finetune.freeze_layers:
            print(f"Freezing layers: {cfg.finetune.freeze_layers}")
            # PLACEHOLDER: Implement layer freezing logic
        
        # Adjust learning rate for fine-tuning
        if hasattr(cfg.finetune, 'learning_rate'):
            print(f"Setting fine-tuning learning rate: {cfg.finetune.learning_rate}")
            # PLACEHOLDER: Implement learning rate adjustment logic
        
        print("Fine-tuning setup complete.")
        
    else:
        print("Setting up model for training from scratch...")
        model = Model(cfg=cfg.architecture)
    
    return model

def detect_mode(cfg):
    """Detect whether we're in training or fine-tuning mode."""
    return (hasattr(cfg, 'finetune') and 
            cfg.finetune.enabled and 
            hasattr(cfg.finetune, 'base_model_path') and 
            cfg.finetune.base_model_path)


@hydra.main(config_name="config")
def main(cfg):
    # Detect training mode
    finetune_mode = detect_mode(cfg)
    mode_str = "fine-tuning" if finetune_mode else "training"
    
    # Generate experiment name and setup output directory
    experiment_name = f"{'finetune' if finetune_mode else 'train'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup output directory
    if hasattr(cfg, 'output_dir'):
        output_dir = Path(cfg.output_dir) / experiment_name
    else:
        output_dir = Path("experiments") / experiment_name
    
    print(f"Starting {mode_str} experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    save_config(cfg, output_dir)
    
    # Set random seed with wrapper options
    random_seed = getattr(cfg, 'random_seed', 9)
    seed_everything(random_seed, workers=True)
    
    # Setup data paths with wrapper configuration support
    if hasattr(cfg, 'data'):
        # Use wrapper data configuration
        train_path = Path(hydra.utils.to_absolute_path(cfg.data.train_path))
        val_path = Path(hydra.utils.to_absolute_path(cfg.data.val_path))
        data_cfg = cfg.data
    else:
        # Use original configuration
        train_path = Path(hydra.utils.to_absolute_path(cfg.train_path))
        val_path = Path(hydra.utils.to_absolute_path(cfg.val_path))
        data_cfg = cfg
    
    # Create data module
    data = DataModule(train_path, val_path, None, data_cfg)
    
    # Create model (with optional fine-tuning)
    model = setup_model(cfg, finetune_mode)
    
    # Setup loggers
    loggers = setup_loggers(cfg, experiment_name, output_dir)
    
    # Setup callbacks
    callbacks = setup_callbacks(cfg, output_dir)

    # Wrapper trainer configuration
    trainer_kwargs = {
        "max_epochs": cfg.epochs,
        "precision": getattr(cfg, 'precision', 32),
        "logger": loggers,
        "callbacks": callbacks,
        "log_every_n_steps": getattr(cfg, 'log_every_n_steps', 50),
        "val_check_interval": getattr(cfg, 'val_check_interval', 1.0),
        "gradient_clip_val": getattr(cfg, 'gradient_clip_val', 1.0),
        "accumulate_grad_batches": getattr(cfg, 'accumulate_grad_batches', 1),
    }
    
    # Handle device configuration (updated for newer PyTorch Lightning)
    if hasattr(cfg, 'gpu') and cfg.gpu > 0:
        trainer_kwargs["devices"] = cfg.gpu
        trainer_kwargs["accelerator"] = "gpu"
    
    # Handle resume from checkpoint
    resume_checkpoint = getattr(cfg, 'resume_from_checkpoint', None)
    if resume_checkpoint:
        trainer_kwargs["resume_from_checkpoint"] = resume_checkpoint

    trainer = pl.Trainer(**trainer_kwargs)
    
    # Start training/fine-tuning
    print(f"Starting {mode_str}...")
    trainer.fit(model, data)
    
    print(f"{mode_str.capitalize()} completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "0"  # ,1,2,3,4,5,6,7"  # ,1,2,4,5,6,7" Change Me
    )
    main()
