import numpy as np


def run_sklearngp(X_train, y_train, X_test, y_test, cfg):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF

    kernel = ConstantKernel() * RBF()
    model = GaussianProcessRegressor(
        n_restarts_optimizer    = cfg["n_restarts_optimizer"],
        kernel                  = kernel
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((y_test - preds)**2)

    is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True
    else:
        correct = False

    return mse, None, correct



def run_sklearngp_noise(X_train, y_train, X_test, y_test, cfg):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    
    noise = np.random.normal(loc=0.0, scale=cfg["noise"]["std"], size=y_test.shape)
    y_train = y_train + noise

    kernel = ConstantKernel() * RBF()
    model = GaussianProcessRegressor(
        n_restarts_optimizer    = cfg["n_restarts_optimizer"],
        kernel                  = kernel
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((y_test - preds)**2)

    is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True
    else:
        correct = False
        
    return mse, None, correct