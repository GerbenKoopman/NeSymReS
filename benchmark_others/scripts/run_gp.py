import numpy as np


def run_gp(X, y, cfg):
    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function


    def protected_exp(x):
        try:
            return np.where(x < 100, np.exp(x), 1e10)  # prevent overflow
        except:
            return np.ones_like(x) * 1e1

    exp_function = make_function(function = protected_exp, name='exp', arity=1)

    model = SymbolicRegressor(
        population_size     = cfg["population_size"],
        tournament_size     = 20,
        p_crossover         = 0.9,
        p_subtree_mutation  = 0.01,
        function_set        = ["add","sub","mul","div","sqrt","log","neg","inv","sin","cos",exp_function],
        const_range         = (-4*np.pi, 4*np.pi),
        verbose = 1
    )

    model.fit(X, y)
    preds = model.predict(X)
    mse = np.mean((y - preds)**2)
    return mse, model._program.__str__()


def run_gp_noise(X, y, cfg):

    noise = np.random.normal(loc=0.0, scale=cfg["noise"]["std"], size=y.shape)
    
    y = y + noise

    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function


    def protected_exp(x):
        try:
            return np.where(x < 100, np.exp(x), 1e10)  # prevent overflow
        except:
            return np.ones_like(x) * 1e1

    exp_function = make_function(function = protected_exp, name='exp', arity=1)

    model = SymbolicRegressor(
        population_size     = cfg["population_size"],
        tournament_size     = 20,
        p_crossover         = 0.9,
        p_subtree_mutation  = 0.01,
        function_set        = ["add","sub","mul","div","sqrt","log","neg","inv","sin","cos",exp_function],
        const_range         = (-4*np.pi, 4*np.pi),
        verbose = 1
    )

    model.fit(X, y)
    preds = model.predict(X)
    mse = np.mean((y - preds)**2)
    return mse, model._program.__str__()