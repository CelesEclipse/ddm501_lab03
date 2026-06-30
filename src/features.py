"""Feature engineering shared by training and serving (identical to Lab 02).

NOTE on training-serving skew: add_features runs OUTSIDE the model, so the API
must call it too (see api/main.py) using the exact same column names.
"""
import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["petal_area"] = X["petal length (cm)"] * X["petal width (cm)"]
    X["sepal_petal_ratio"] = X["sepal length (cm)"] / (X["petal length (cm)"] + 1e-6)
    return X
