import numpy as np
import tempfile
import json
import numpy as np
from dso import DeepSymbolicRegressor



def run_dsr(X_train, y_train, X_test, y_test, cfg):

    model = DeepSymbolicRegressor(cfg)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = np.mean((y_test - preds)**2)

    is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True
    else:
        correct = False
    
    return mse, model.program_.pretty(), correct



def run_dsr_noise(X_train, y_train, X_test, y_test, cfg):

    noise = np.random.normal(loc=0.0, scale=cfg["noise"]["std"], size=y_test.shape)
    y_test = y_test + noise

    model = DeepSymbolicRegressor(tmp_path)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = np.mean((y_test - preds)**2)

    is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True
    else:
        correct = False
    
    return mse, model.program_.pretty(), correct