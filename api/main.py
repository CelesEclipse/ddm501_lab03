"""FastAPI service for the Iris classifier.

Model loading is backward-compatible with Lab 02:
  - if MODEL_URI is set (e.g. models:/iris-classifier/Production) load from the
    MLflow registry, exactly as in Lab 2;
  - otherwise load a local model.joblib, so tests (Lab 3) and the local stack
    (Lab 4) run with no MLflow server. The load is lazy.
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import add_features

MODEL_URI = os.environ.get("MODEL_URI")               # set this for the registry path
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

app = FastAPI(title="Iris Classifier")
_model = None


def get_model():
    global _model
    if _model is None:
        if MODEL_URI:
            import mlflow.pyfunc
            _model = mlflow.pyfunc.load_model(MODEL_URI)
        else:
            _model = joblib.load(MODEL_PATH)
    return _model


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    X = pd.DataFrame([{
        "sepal length (cm)": features.sepal_length,
        "sepal width (cm)": features.sepal_width,
        "petal length (cm)": features.petal_length,
        "petal width (cm)": features.petal_width,
    }])
    X = add_features(X)
    return {"prediction": int(get_model().predict(X)[0])}
