"""Dynamic linear regression with time-varying coefficients.
Outputs
-------
artifacts/models/dynamic_reg.pkl
artifacts/forecasts/dynamic_reg_test.csv
artifacts/metrics/advanced_classical_metrics.csv (appends)
"""
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM = ROOT / "artifacts" / "models"
ARTF = ROOT / "artifacts" / "forecasts"
METR = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METR):
    p.mkdir(parents=True, exist_ok=True)

# load data
full = pd.read_parquet(DATA).dropna(subset=["log_ret"])

y = full["log_ret"]
X = full.drop(columns=["log_ret"])

train = y.loc[: "2023-12-31"]
train_X = X.loc[: "2023-12-31"]

test = y.loc["2024-01-01":]
test_X = X.loc["2024-01-01":]

# dynamic regression: local level + time-varying betas
model = SARIMAX(
    train,
    exog=train_X,
    order=(0, 0, 0),
    trend="n",
    time_varying_regression=True,
    mle_regression=False,
)
fit = model.fit(disp=False)

joblib.dump(fit, ARTM / "dynamic_reg.pkl")

forecast = fit.get_forecast(steps=len(test), exog=test_X)
pred = forecast.predicted_mean
pred.to_csv(ARTF / "dynamic_reg_test.csv")

# metrics
mae = np.mean(np.abs(test - pred))
rmse = np.sqrt(np.mean((test - pred) ** 2))

eps = 1e-8
true_vals = test.values
denom = np.where(np.abs(true_vals) < eps, eps, true_vals)
mape = np.mean(np.abs((true_vals - pred.values) / denom)) * 100

row = pd.DataFrame({"model": ["DynamicReg"], "MAE": [mae], "RMSE": [rmse], "MAPE": [mape]})
met_file = METR / "advanced_classical_metrics.csv"
if met_file.exists():
    row.to_csv(met_file, mode="a", header=False, index=False)
else:
    row.to_csv(met_file, index=False)

print("Saved model →", ARTM / "dynamic_reg.pkl")
print("Saved forecast →", ARTF / "dynamic_reg_test.csv")
