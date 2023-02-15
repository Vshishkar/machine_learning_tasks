import numpy as np
import matplotlib.pyplot as plt

def zscore_normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)

    data_norm = zscore_normalize_given_params(data, sigma, mu)

    return (data_norm, sigma, mu)

def zscore_normalize_given_params(data, sigma, mu):
    data_norm = (data - mu) / sigma
    return data_norm

def plot_feature_against_price(x, y, feature_label, y_label):
    ax, fig = plt.subplots()
    fig.scatter(x, y)

    plt.ylabel(y_label)
    plt.xlabel(feature_label)
    plt.show()


def plot_all_features(X, y):
    for i, feature_column in enumerate(np.transpose(X)):
        plot_feature_against_price(feature_column, y, "test", "Test")

def plot_cost(data):
    fig, ax = plt.subplots()

    x = [x for _, x in data]
    y = [y for y, _ in data]

    ax.plot(x, y)
    plt.xlabel("iterations")
    plt.ylabel("cost")

    plt.show()


def plot_prediction_and_data(x1, y1, x2, y2):
    fig, ax = plt.subplots()

    ax.plot(x1, y1, color="tab:blue")
    ax.plot(x2, y2, color="tab:orange")

    plt.show()