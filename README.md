# Lab 03 — Unit Testing & CI/CD

Uses the same Iris contract as Labs 1–2 (sklearn column names, `species` target, `add_features` -> `petal_area` + `sepal_petal_ratio`, model = `Pipeline(StandardScaler, RandomForest)`).

The testing scope is unit tests (feature engineering + request schema). The focus of the lab is wiring those unit tests into Continuous Integration on a self-hosted GitHub Actions runner on your own machine.

## Layout
```
lab03/
├── src/
│   ├── data.py            # load_raw()  (sklearn columns + species)
│   └── features.py        # add_features()  (identical to Lab 02)
├── api/main.py            # FastAPI: /health, /predict, IrisFeatures
├── train.py              # Pipeline(StandardScaler, RandomForest) -> model.joblib
├── tests/
│   ├── test_features.py   # UNIT - feature engineering
│   └── test_schema.py     # UNIT - request schema validation
├── .github/workflows/ci.yml   # runs-on: self-hosted
├── pytest.ini             # pythonpath = . (so `src`/`api` import cleanly)
├── requirements.txt       # pinned (scikit-learn==1.4.2, pandas==2.2.2, ...)
└── requirements-dev.txt   # pytest
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -v          # expected: 8 passed
```