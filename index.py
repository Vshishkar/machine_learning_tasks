from linear_regression.data_processor import *
from linear_regression.linear_regression import *


def main():
    X_train, X_test, y_train, y_test = load_data()
    linear_regression(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()