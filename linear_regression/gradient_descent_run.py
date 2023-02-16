import numpy as np
import linear_regression.cost_function as cf
import linear_regression.compute_gradient as cg
import linear_regression.utils as ut

def gradient_descent_run(X, y):
    m, n = X.shape

    w = np.zeros(n)
    b = 0

    lambda_ = 1

    iterations = 0
    epsilon = 10**-3

    alpha = 0.01

    previousCost = float('inf')
    currentCost = cf.compute_cost(X, w, b, y, lambda_)

    cost_history = [(currentCost, 0)]

    while previousCost - currentCost > epsilon or iterations < 1000:
        dj_dw, dj_db = cg.compute_gradient(X, w, b, y, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        previousCost = currentCost
        currentCost = cf.compute_cost(X, w, b,  y, lambda_)

        cost_history.append((currentCost, iterations))
        iterations += 1
        if iterations % 500 == 0:
            # plot graph
            ut.plot_cost(cost_history)
    return w, b