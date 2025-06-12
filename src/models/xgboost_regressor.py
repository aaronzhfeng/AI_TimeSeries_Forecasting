"""
XGBoost regressor for NASDAQ forecasting.
Saves:
  artifacts/models/xgboost.model
  artifacts/forecasts/xgboost_test.csv
  artifacts/metrics/ml_regressor_metrics.csv (appends)
"""
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

MAX_LAG = 30
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM = ROOT / "artifacts" / "models"
ARTF = ROOT / "artifacts" / "forecasts"
METR = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------

def make_features(df: pd.DataFrame, target: str = "log_ret", lags: int = MAX_LAG):
    cols = []
    for i in range(1, lags + 1):
        shifted = df.shift(i).add_suffix(f"_lag{i}")
        cols.append(shifted)
    lagged = pd.concat(cols, axis=1)
    lagged[target] = df[target]
    lagged = lagged.dropna()
    X = lagged.drop(columns=[target])
    y = lagged[target]
    return X, y

# load data
full = pd.read_parquet(DATA)
X, y = make_features(full)
train_idx = y.index < "2024-01-01"
X_train, X_test = X.loc[train_idx], X.loc[~train_idx]
y_train, y_test = y.loc[train_idx], y.loc[~train_idx]

# search
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}
model = XGBRegressor(objective="reg:squarederror", verbosity=0)
search = GridSearchCV(model, param_grid, cv=tscv, n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_model.save_model(str(ARTM / "xgboost.model"))

pred = best_model.predict(X_test)
pd.Series(pred, index=X_test.index).to_csv(ARTF / "xgboost_test.csv")

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

eps = 1e-8
true_vals = y_test.values
denom = np.where(np.abs(true_vals) < eps, eps, true_vals)
mape = np.mean(np.abs((true_vals - pred) / denom)) * 100

row = pd.DataFrame({"model": ["XGBoost"], "MAE": [mae], "RMSE": [rmse], "MAPE": [mape]})
met_file = METR / "ml_regressor_metrics.csv"
if met_file.exists():
    row.to_csv(met_file, mode="a", header=False, index=False)
else:
    row.to_csv(met_file, index=False)

print("Saved model →", ARTM / "xgboost.model")
print("Saved forecast →", ARTF / "xgboost_test.csv")

