"""Bayesian structural time series using Prophet.
Outputs
-------
artifacts/models/prophet.pkl
artifacts/forecasts/prophet_test.csv
artifacts/metrics/advanced_classical_metrics.csv (appends)
"""
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM = ROOT / "artifacts" / "models"
ARTF = ROOT / "artifacts" / "forecasts"
METR = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METR):
    p.mkdir(parents=True, exist_ok=True)

# load data and rename
full = pd.read_parquet(DATA).dropna(subset=["log_ret"])
full = full.reset_index().rename(columns={"index": "ds", "log_ret": "y"})

train = full[full["ds"] <= "2023-12-31"]
test = full[full["ds"] >= "2024-01-01"]

m = Prophet()
for col in train.columns:
    if col not in ["ds", "y"]:
        m.add_regressor(col)

m.fit(train)

future = test.drop(columns="y")
forecast = m.predict(future)
forecast["yhat"].to_csv(ARTF / "prophet_test.csv", index=test["ds"])

joblib.dump(m, ARTM / "prophet.pkl")

# metrics
pred = forecast["yhat"].values
true = test["y"].values
mae = np.mean(np.abs(true - pred))
rmse = np.sqrt(np.mean((true - pred) ** 2))

eps = 1e-8
denom = np.where(np.abs(true) < eps, eps, true)
mape = np.mean(np.abs((true - pred) / denom)) * 100

row = pd.DataFrame({"model": ["Prophet"], "MAE": [mae], "RMSE": [rmse], "MAPE": [mape]})
met_file = METR / "advanced_classical_metrics.csv"
if met_file.exists():
    row.to_csv(met_file, mode="a", header=False, index=False)
else:
    row.to_csv(met_file, index=False)

print("Saved model →", ARTM / "prophet.pkl")
print("Saved forecast →", ARTF / "prophet_test.csv")
