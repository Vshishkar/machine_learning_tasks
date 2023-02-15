import numpy as np

def compute_gradient(X, w, b, y, lambda_):
    m, n = X.shape

    cost = np.matmul(X, w) + b - y

    dj_dw = np.matmul(np.transpose(cost), X) / m
    dj_db = sum(cost) / m

    dj_dw += lambda_ * w / m

    return dj_dw, dj_db
