"""
SARIMA grid-search with tqdm progress bar (weekly seasonality).

Artifacts
---------
artifacts/models/sarima.pkl
artifacts/forecasts/sarima_test.csv
artifacts/residuals/sarima_residuals.csv
"""
import itertools, gc, joblib, numpy as np, pandas as pd, statsmodels.api as sm
from tqdm import tqdm
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "nasdaq_features.parquet"
OUTM = ROOT / "artifacts" / "models"     / "sarima.pkl"
OUTF = ROOT / "artifacts" / "forecasts"  / "sarima_test.csv"
OUTR = ROOT / "artifacts" / "residuals"  / "sarima_residuals.csv"
for p in (OUTM.parent, OUTF.parent, OUTR.parent):
    p.mkdir(parents=True, exist_ok=True)

# ── series split ─────────────────────────────────────────────────────────
y = pd.read_parquet(DATA)["log_ret"].dropna()
train, test = y.loc[: "2023-12-31"], y.loc["2024-01-01":]

# ── search grid ─────────────────────────────────────────────────────────
p = q = range(0, 3)       # 0,1,2
P = Q = range(0, 2)       # 0,1
seasonal_period = 5       # weekly (5 trading days)

best_aic  = np.inf
best_mod  = None
best_cfg  = None

grid = list(itertools.product(p, q, P, Q))
for (p_, q_, P_, Q_) in tqdm(grid, desc="SARIMA grid"):
    try:
        model = sm.tsa.statespace.SARIMAX(
            train,
            order=(p_, 0, q_),
            seasonal_order=(P_, 1, Q_, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        if model.aic < best_aic:
            best_aic, best_mod = model.aic, model
            best_cfg = (p_, 0, q_, P_, 1, Q_, seasonal_period)
    except (ValueError, np.linalg.LinAlgError):
        pass
    gc.collect()  # free memory after each fit

if best_mod is None:
    raise RuntimeError("All SARIMA fits failed. Try expanding the grid.")

print("Best SARIMA:", best_cfg, "AIC=", best_aic)

# ── save model ───────────────────────────────────────────────────────────
joblib.dump(best_mod, OUTM)

# ── forecasts & residuals ────────────────────────────────────────────────
in_sample = best_mod.get_prediction(
    start=train.index[0],
    end=train.index[-1]
).predicted_mean
forecast = best_mod.get_forecast(steps=len(test)).predicted_mean
pd.Series(forecast, index=test.index).to_csv(OUTF)
(train - in_sample).to_csv(OUTR)

print("Artifacts written:")
print("  •", OUTM.relative_to(ROOT))
print("  •", OUTF.relative_to(ROOT))
print("  •", OUTR.relative_to(ROOT))

