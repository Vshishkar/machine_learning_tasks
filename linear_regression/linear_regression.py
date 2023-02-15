from gradient_descent_run import *

def linear_regression(X_train, y_train, X_test, y_test):
    # normalize data
    X_train, sigma, mu = zscore_normalize(X_train)

    # plot normalized data

    m, n = X_train.shape
    y_train = y_train.reshape(m)
    y_test = y_test.reshape(len(y_test))
    plot_all_features(X_train, y_train)
    # do gradient descent
    w, b = gradient_descent_run(X_train, y_train)
    # plot gradient descent against iterations

    # converge
    X_test = zscore_normalize_given_params(X_test, sigma, mu)
    predict = calc_predict(X_test, w, b)
    print(np.mean(abs(y_test - predict)))


def calc_predict(X, w, b):
    return np.matmul(X, w) + b