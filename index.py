import linear_regression.data_processor as d_pr
import linear_regression.linear_regression as lr
import sys

def main():
    X_train, X_test, y_train, y_test = d_pr.load_data()
    lr.linear_regression(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()