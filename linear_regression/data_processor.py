import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    dataset = pd.read_csv("data/data.csv")
    dataset.drop(['id', 'date', "zipcode"], axis=1)

    X_labels = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
                "sqft_above", "sqft_basement", "yr_built", "yr_renovated",  "lat", "long", "sqft_living15", "sqft_lot15"]
    y_label = "price"

    X = dataset.loc[:, ["sqft_living"]].values
    y = dataset.loc[:, [y_label]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test