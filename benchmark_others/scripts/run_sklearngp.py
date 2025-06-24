import numpy as np

def run_sklearngp(X, y, cfg):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF

    kernel = ConstantKernel() * RBF()
    model = GaussianProcessRegressor(
        n_restarts_optimizer    = cfg["n_restarts_optimizer"]
    )

    model.fit(X, y)
    preds = model.predict(X)
    mse = np.mean((y - preds)**2)

    return mse