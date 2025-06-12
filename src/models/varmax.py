"""VARMAX model for joint NASDAQ and macro series.
Outputs
-------
artifacts/models/varmax.pkl
artifacts/forecasts/varmax_test.csv
artifacts/metrics/advanced_classical_metrics.csv (appends)
"""
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.varmax import VARMAX
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM = ROOT / "artifacts" / "models"
ARTF = ROOT / "artifacts" / "forecasts"
METR = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METR):
    p.mkdir(parents=True, exist_ok=True)

# load and select columns
full = pd.read_parquet(DATA).dropna()
cols = full.columns[:4]  # log_ret + first 3 features
series = full[cols]

train = series.loc[: "2023-12-31"]
test = series.loc["2024-01-01":]

model = VARMAX(train, order=(1, 0))
fit = model.fit(disp=False)

joblib.dump(fit, ARTM / "varmax.pkl")

forecast = fit.forecast(steps=len(test))
forecast.index = test.index
forecast["log_ret"].to_csv(ARTF / "varmax_test.csv")

# metrics
true_vals = test["log_ret"]
pred_vals = forecast["log_ret"]
mae = np.mean(np.abs(true_vals - pred_vals))
rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))

eps = 1e-8
denom = np.where(np.abs(true_vals) < eps, eps, true_vals)
mape = np.mean(np.abs((true_vals - pred_vals) / denom)) * 100

row = pd.DataFrame({"model": ["VARMAX"], "MAE": [mae], "RMSE": [rmse], "MAPE": [mape]})
met_file = METR / "advanced_classical_metrics.csv"
if met_file.exists():
    row.to_csv(met_file, mode="a", header=False, index=False)
else:
    row.to_csv(met_file, index=False)

print("Saved model →", ARTM / "varmax.pkl")
print("Saved forecast →", ARTF / "varmax_test.csv")
