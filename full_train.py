#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
import time

def create_config():
    cfg = OmegaConf.create({
        "epochs": 20,
        "batch_size": 16,
        "num_of_workers": 4,
        "precision": 16,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 2,
        "num_of_points": 800,
        "activation_name": "gelu",
        "activation_token": 7,
        "positional_embedding_num": 30,
        "linear": False,
        "bit16": True,
        "norm": True,
        "mean": 0.5,
        "std": 0.5,
        "activation": "relu",
        "input_normalization": False,
        "dim_input": 4,
        "dim_hidden": 512,
        "num_heads": 8,
        "num_inds": 50,
        "ln": True,
        "n_l_enc": 6,
        "num_features": 10,
        "sinuisodal_embeddings": False,
        "trg_pad_idx": 0,
        "src_pad_idx": 0,
        "output_dim": 60,
        "length_eq": 60,
        "dec_pf_dim": 512,
        "dec_layers": 6,
        "dropout": 0.1,
        "lr": 0.0003,
        "num_encoder_layer": 8,
        "hidden_size": 512,
        "num_hidden_layer_encoder": 3,
        "ffn_hidden_size": 2048,
        "num_decoder_layer": 8,
        "num_hidden_layer_decoder": 3,

        "dataset_train": {
            "max_number_of_points": 500,
            "type_of_sampling_points": "logarithm",
            "predict_c": True,
            "fun_support": {"max": 10, "min": -10},
            "constants": {
                "num_constants": 3,
                "additive": {"max": 5, "min": -5},
                "multiplicative": {"max": 3, "min": -3}
            }
        }
    })

    cfg.dataset_val = cfg.dataset_train
    cfg.dataset_test = cfg.dataset_train

    return cfg

def setup_callbacks(output_dir):
    """Setup training callbacks"""

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="nesymres-{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}",
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_weights_only=False
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop_callback)

    return callbacks

def setup_loggers(output_dir):
    """Setup comprehensive logging"""

    loggers = []

    csv_logger = CSVLogger(
        save_dir=output_dir / "logs",
        name="nesymres_training"
    )
    loggers.append(csv_logger)

    tb_logger = TensorBoardLogger(
        save_dir=output_dir / "tensorboard",
        name="nesymres_training"
    )
    loggers.append(tb_logger)

    return loggers

def generate_data(train_samples=1000, val_samples=200):
    """Generate larger datasets for training"""

    print(f"Generating datasets...")
    print(f"  Training samples: {train_samples}")
    print(f"  Validation samples: {val_samples}")

    train_path = Path("data/raw_datasets/train")
    train_path.mkdir(parents=True, exist_ok=True)

    result = os.system(f"cd scripts/data_creation && python dataset_creation.py {train_samples} ../../{train_path}")
    if result != 0:
        print(f"Failed to generate training data")
        return False

    val_path = Path("data/raw_datasets/val")
    val_path.mkdir(parents=True, exist_ok=True)

    result = os.system(f"cd scripts/data_creation && python dataset_creation.py {val_samples} ../../{val_path}")
    if result != 0:
        print(f"Failed to generate validation data")
        return False

    print(f"Datasets generated")
    return train_path, val_path

def main():
    """Main training function"""

    print("NESYMRES Training")
    print("=" * 60)

    try:

        print("Importing NESYMRES components...")
        from nesymres.architectures.model import Model
        from nesymres.architectures.data import DataModule

        print("All imports successful!")

        print("\nCreating configuration...")
        cfg = create_config()
        print("Configuration created")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/training_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        OmegaConf.save(cfg, output_dir / "config.yaml")

        print("\nSetting up datasets...")

        large_train_path = Path("data/raw_datasets/100")
        large_val_path = Path("data/raw_datasets/50")

        if large_train_path.exists() and large_val_path.exists():
            train_files = list(large_train_path.glob("*.h5"))
            val_files = list(large_val_path.glob("*.h5"))
            print(f"Using existing datasets: {len(train_files)} train, {len(val_files)} val files")
            train_path = large_train_path
            val_path = large_val_path
        else:

            result = generate_data(1000, 200)
            if not result:
                return False
            train_path, val_path = result

        print("\nCreating data module...")
        data = DataModule(
            data_train_path=train_path,
            data_val_path=val_path,
            data_test_path=val_path,
            cfg=cfg
        )
        print("Data module created")

        print("\nCreating model...")
        model = Model(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params/1e6:.1f}M parameters")

        print("\nSetting up training infrastructure...")
        callbacks = setup_callbacks(output_dir)
        loggers = setup_loggers(output_dir)
        print("Callbacks and loggers configured")

        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
            print(f"Using GPU acceleration")
        else:
            accelerator = "cpu"
            devices = 1
            print("Using CPU (consider GPU for faster training)")

        print("\nSetting up trainer...")
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            precision=cfg.precision,
            logger=loggers,
            callbacks=callbacks,
            log_every_n_steps=10,
            val_check_interval=0.5,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=cfg.gradient_clip_val,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            num_sanity_val_steps=2,

        )
        print("Trainer configured")

        print("\nStarting training...")
        print("\nConfiguration:")
        print(f" - Epochs: {cfg.epochs}")
        print(f" - Batch size: {cfg.batch_size} (effective: {cfg.batch_size * cfg.accumulate_grad_batches})")
        print(f" - Learning rate: {cfg.lr}")
        print(f" - Model size: ~{total_params/1e6:.1f}M parameters")
        print(f" - Max data points: {cfg.dataset_train.max_number_of_points}")
        print(f" - Constants: {cfg.dataset_train.constants.num_constants}")
        print(f" - Precision: {cfg.precision}-bit")
        print(f" - Accelerator: {accelerator}")
        print(f" - Workers: {cfg.num_of_workers}")
        print(f" - Validation: Every 0.5 epochs")
        print(f" - Early stopping: 5 epochs patience")

        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)

        start_time = time.time()
        trainer.fit(model, data)
        end_time = time.time()

        training_time = end_time - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)

        print(f"\nTraining Results:")
        print(f" - Total time: {training_time/3600:.2f} hours")
        print(f" - Epochs completed: {trainer.current_epoch + 1}")
        print(f" - Best model checkpoint: {callbacks[0].best_model_path}")

        if hasattr(trainer, 'callback_metrics'):
            metrics = trainer.callback_metrics
            if 'train_loss' in metrics:
                print(f" - Final training loss: {metrics['train_loss']:.4f}")
            if 'val_loss' in metrics:
                print(f" - Final validation loss: {metrics['val_loss']:.4f}")

        print(f"\nResults saved to: {output_dir}")
        print(f" - Model checkpoints: {output_dir}/checkpoints/")
        print(f" - Training logs: {output_dir}/logs/")
        print(f" - TensorBoard logs: {output_dir}/tensorboard/")
        print(f" - Configuration: {output_dir}/config.yaml")

        print("\nTraining completed successfully!")
        print("Model is ready for inference and evaluation")

        return True

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("TRAINING SUCCESSFUL!")
        print("NESYMRES model trained on data")
        print("Ready for deployment and evaluation")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED")
        print("Check error messages above")
        print("=" * 60)
        sys.exit(1)
