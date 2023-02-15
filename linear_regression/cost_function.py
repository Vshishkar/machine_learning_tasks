import numpy as np

def compute_cost(X, w, b, y, lambda_):
    m = len(y)

    total_cost = 0
    cost = np.matmul(X, w) + b
    cost = cost - y

    total_cost = np.dot(cost, cost) * 0.5 / m + \
        lambda_ * np.dot(w, w) * 0.5 / m

    return total_cost