import numpy as np


def run_gp(X_train, y_train, X_test, y_test, cfg):
    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function

    # Have to manually define exp, default doesn't come with it
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

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((y_test - preds)**2)

    is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True
    else:
        correct = False

    return mse, model._program.__str__(), correct


def run_gp_noise(X_train, y_train, X_test, y_test, cfg):

    noise = np.random.normal(loc=0.0, scale=cfg["noise"]["std"], size=y_test.shape)
    y_test = y_test + noise

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

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((y_test - preds)**2)

    is_close = np.isclose(y_test, preds, rtol=0.05, atol=1e-3)
    accuracy = np.mean(is_close)
    if accuracy >= 0.95:
        correct = True
    else:
        correct = False

    return mse, model._program.__str__(), correct