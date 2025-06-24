#!/usr/bin/env python3
"""
Simple NESYMRES test script - tests basic model loading and inference without Hydra.
This is a standalone validation script for the unified workflow.
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

def find_model_weights():
    """Find available model weights."""
    weights_dir = Path(__file__).parent.parent / "weights"
    available_models = []
    
    for model_file in weights_dir.glob("*.ckpt"):
        available_models.append(model_file)
    
    return available_models

def test_model_loading():
    """Test basic model loading and setup."""
    print("Testing NESYMRES Model Loading (Unified Workflow)")
    print("=" * 50)
    
    from nesymres.architectures.model import Model
    from nesymres.dclasses import BFGSParams, FitParams
    
    # Find available model weights
    available_models = find_model_weights()
    
    if not available_models:
        print("ERROR: No model weights found in weights/ directory")
        print("   Please ensure you have model checkpoints available")
        return False
    
    # Use the first available model
    model_path = available_models[0]
    print(f"Loading model from: {model_path}")
    print(f"Available models: {[m.name for m in available_models]}")
    
    # Create a simple config for the architecture
    class SimpleConfig:
        def __init__(self):
            # Default architecture parameters from the original config
            self.sinuisodal_embeddings = False
            self.dec_pf_dim = 512
            self.dec_layers = 5
            self.dim_hidden = 512
            self.lr = 0.0001
            self.dropout = 0
            self.num_features = 10
            self.ln = True
            self.N_p = 0
            self.num_inds = 50
            self.activation = "relu"
            self.bit16 = True
            self.norm = True
            self.linear = False
            self.input_normalization = False
            self.src_pad_idx = 0
            self.trg_pad_idx = 0
            self.length_eq = 60
            self.n_l_enc = 5
            self.mean = 0.5
            self.std = 0.5
            self.dim_input = 4
            self.num_heads = 8
            self.output_dim = 60
    
    cfg = SimpleConfig()
    
    try:
        # Load model
        model = Model.load_from_checkpoint(model_path, cfg=cfg)
        model.eval()
        print("Model loaded successfully")
        
        if torch.cuda.is_available():
            model.cuda()
            print("Model moved to GPU")
        else:
            print("WARNING: Running on CPU")
        
        # Test basic prediction setup
        print("\\nSetting up prediction parameters...")
        
        # Create minimal test configuration
        class DummyTestData:
            def __init__(self):
                # Use the correct vocabulary from eq_setting.json
                self.word2id = {
                    "P": 0, "S": 1, "F": 2, "c": 3,
                    "x_1": 4, "x_2": 5, "x_3": 6,
                    "abs": 7, "acos": 8, "add": 9, "asin": 10, "atan": 11,
                    "cos": 12, "cosh": 13, "coth": 14, "div": 15, "exp": 16,
                    "ln": 17, "mul": 18, "pow": 19, "sin": 20, "sinh": 21,
                    "sqrt": 22, "tan": 23, "tanh": 24,
                    "-3": 25, "-2": 26, "-1": 27, "0": 28, "1": 29, "2": 30, "3": 31, "4": 32, "5": 33
                }
                self.id2word = {v: k for k, v in self.word2id.items()}
                self.una_ops = ["asin", "cos", "exp", "ln", "pow2", "pow3", "pow4", "pow5", "sin", "sqrt", "tan"]
                self.bin_ops = ["add", "mul", "div", "pow"]  # corrected binary ops
                self.total_variables = ["x_1", "x_2", "x_3"]
                self.total_coefficients = ["cm_0", "cm_1", "cm_2", "ca_0", "ca_1", "ca_2"]  # simplified
                self.rewrite_functions = []
        
        test_data = DummyTestData()
        
        # Setup BFGS parameters
        bfgs = BFGSParams(
            activated=True,
            n_restarts=10,
            add_coefficients_if_not_existing=True,
            normalization_o=False,
            idx_remove=True,
            normalization_type="standard",
            stop_time=999999,
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
            beam_size=5
        )
        
        print("Parameters configured")
        
        # Test with simple data
        print("\\nTesting with simple equation: x_1 + x_2")
        
        # Generate simple test data: y = x1 + x2
        np.random.seed(42)
        num_points = 100
        x1 = np.random.uniform(-5, 5, num_points)
        x2 = np.random.uniform(-5, 5, num_points)
        y = x1 + x2
        X = np.column_stack([x1, x2])  # Shape: (100, 2)
        
        print(f"Generated test data: X shape {X.shape}, y shape {y.shape}")
        
        # Create fitfunc
        fitfunc = partial(model.fitfunc, cfg_params=params_fit)
        print("Fitfunc created")
        
        # Test prediction
        print("Running prediction...")
        try:
            results = fitfunc(X, y)
            print("Prediction completed!")
            
            if results and len(results) > 0:
                # results is a list of dictionaries, each with 'config' and 'output'
                first_result = results[0]
                output = first_result.get('output')
                
                if output and 'best_bfgs_preds' in output:
                    pred_equations = output['best_bfgs_preds']
                    print(f"Predicted equations: {pred_equations}")
                    
                    if pred_equations and len(pred_equations) > 0:
                        print(f"Best prediction: {pred_equations[0]}")
                        return True
                    else:
                        print("WARNING: No equations predicted")
                        return False
                else:
                    print("WARNING: Unexpected output format")
                    if output:
                        print(f"Output keys: {list(output.keys())}")
                    else:
                        print("Output is None")
                    return False
            else:
                print("WARNING: No results returned")
                return False
                
        except Exception as e:
            print(f"ERROR: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"ERROR: Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_equations():
    """Test with equations from the CSV file."""
    print("\\nTesting with CSV equations")
    print("=" * 40)
    
    csv_path = "/home/gerben-koopman/studie/nesymres/test_set/nc.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} equations from CSV")
    print("First few equations:")
    for i in range(min(3, len(df))):
        print(f"  {i}: {df.iloc[i]['eq']}")
    
    return True

if __name__ == "__main__":
    print("NESYMRES Unified Workflow - Simple Test")
    print("=" * 50)
    print("This validates that the core NESYMRES functionality works correctly")
    print("after the wrapper features have been integrated into the main scripts.")
    print()
    
    success1 = test_model_loading()
    success2 = test_csv_equations()
    
    if success1 and success2:
        print("\nAll tests passed! The unified NESYMRES workflow is ready.")
        print("\nNext steps:")
        print("1. Use scripts/train.py for training (supports fine-tuning)")
        print("2. Use scripts/evaluate.py for comprehensive evaluation") 
        print("3. Use scripts/fitfunc.py for equation fitting")
        print("4. Check UNIFIED_WORKFLOW.md for detailed usage")
        print("5. Use config.yaml.new as your unified configuration")
    else:
        print("\nSome tests failed. Check the errors above.")
