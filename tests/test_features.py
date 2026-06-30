"""Unit tests for feature engineering (Lab 02 column names)."""
import pandas as pd

from src.features import add_features

ROW = {
    "sepal length (cm)": 5.0,
    "sepal width (cm)": 3.0,
    "petal length (cm)": 1.5,
    "petal width (cm)": 0.2,
}


def test_add_features_creates_columns():
    out = add_features(pd.DataFrame([ROW]))
    assert "petal_area" in out.columns
    assert "sepal_petal_ratio" in out.columns


def test_add_features_computes_correct_values():
    out = add_features(pd.DataFrame([ROW]))
    assert out.loc[0, "petal_area"] == 1.5 * 0.2
    assert out.loc[0, "sepal_petal_ratio"] == 5.0 / (1.5 + 1e-6)


def test_add_features_does_not_mutate_input():
    df = pd.DataFrame([ROW])
    add_features(df)
    assert "petal_area" not in df.columns


def test_add_features_preserves_row_count():
    df = pd.DataFrame([ROW, ROW, ROW])
    out = add_features(df)
    assert len(out) == 3
    assert "petal_area" in out.columns
