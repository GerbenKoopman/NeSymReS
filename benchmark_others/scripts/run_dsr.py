import numpy as np
import tempfile
import json
import numpy as np
from dso import DeepSymbolicRegressor


def run_dsr(X_train,
            y_train,
            X_test,
            y_test):

    # Create a temporary JSON config file
    config = {
        "task": {
            "task_type": "regression",
            "dataset": None,
            "function_set": ["add","sub","mul","div","sin","cos","exp","log", "const"],
        },

        "policy_optimizer" : {
            "learning_rate" : 0.0005,
            "entropy_weight" : 0.005,
            "entropy_gamma" : 0.7

        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp)
        tmp_path = tmp.name

    try:
        model = DeepSymbolicRegressor(config_dict=tmp_path)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = np.mean((y_test - preds)**2)

        is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
        accuracy = np.mean(is_close)
        if accuracy >= 0.95:
            correct = True
        else:
            correct = False
    
    finally:
        # Clean up temp config file
        try:
            import os
            os.remove(tmp_path)
        except OSError:
            pass

    return mse, model.program_.pretty(), correct



def run_dsr_noise(X_train,
            y_train,
            X_test,
            y_test):

    # Create a temporary JSON config file
    config = {
        "task": {
            "task_type": "regression",
            "dataset": None,
            "function_set": ["add","sub","mul","div","sin","cos","exp","log", "const"],
        },

        "policy_optimizer" : {
            "learning_rate" : 0.0005,
            "entropy_weight" : 0.005,
            "entropy_gamma" : 0.7

        }
    }

    noise = np.random.normal(loc=0.0, scale=0.1, size=y_test.shape)
    y_test = y_test + noise

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp)
        tmp_path = tmp.name

    try:
        model = DeepSymbolicRegressor(config_dict=tmp_path)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = np.mean((y_test - preds)**2)

        is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
        accuracy = np.mean(is_close)
        if accuracy >= 0.95:
            correct = True
        else:
            correct = False
    
    finally:
        # Clean up temp config file
        try:
            import os
            os.remove(tmp_path)
        except OSError:
            pass

    return mse, model.program_.pretty(), correct