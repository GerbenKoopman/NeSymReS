import numpy as np


def run_dsr(X, y, cfg):
    from dso import DeepSymbolicRegressor as DSR

    model = DSR(cfg)

    model.fit(X, y)

    preds = model.predict(X)
    mse   = float(np.mean((y - preds) ** 2))

    expr = model.program_.pretty()

    is_close = np.isclose(y, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True

    return mse, expr, correct