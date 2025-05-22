"""
ARIMA & ARIMAX training script
Usage:  python src/models/arima.py
"""
import pandas as pd, joblib, json, warnings, numpy as np
from pmdarima import auto_arima
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM   = ROOT / "artifacts" / "models"
ARTF   = ROOT / "artifacts" / "forecasts"
ARTR   = ROOT / "artifacts" / "residuals"
for p in (ARTM, ARTF, ARTR):
    p.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# ── load & train/test split ──────────────────────────────────────────────
df = pd.read_parquet(DATA).dropna(subset=["log_ret"])
train = df.loc[: "2023-12-31"]
test  = df.loc["2024-01-01":]

y_train, y_test = train["log_ret"], test["log_ret"]
X_train, X_test = train.drop(columns="log_ret"), test.drop(columns="log_ret")

# ── ARIMA (no exogenous) ─────────────────────────────────────────────────
arima = auto_arima(
    y=y_train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)
arima_fit = arima.fit(y_train)
joblib.dump(arima_fit, ARTM / "arima.pkl")

fc_arima = arima_fit.predict_in_sample()
fcast_arima = arima_fit.predict(n_periods=len(y_test))
pd.Series(fcast_arima, index=y_test.index).to_csv(ARTF / "arima_test.csv")

res_arima = y_train - fc_arima
res_arima.to_csv(ARTR / "arima_residuals.csv")

print("ARIMA done:", arima_fit.order)

# ── CLEAN EXOGENEOUS MATRIX -----------------------------------------------
def clean_exog(X: pd.DataFrame, y: pd.Series):
    X = (
        X.replace([np.inf, -np.inf], np.nan)  # inf → NaN
          .ffill()                            # forward-fill
          .bfill()                            # back-fill (for leading NaNs)
    )
    # drop any rows that *still* have NaN (rare)
    mask = X.notna().all(axis=1)
    return X.loc[mask], y.loc[mask]

X_train_clean, y_train_clean = clean_exog(X_train, y_train)
X_test_clean , y_test_clean  = clean_exog(X_test , y_test)


# ── ARIMAX (macro exogenous) ─────────────────────────────────────────────
arimax = auto_arima(
    y=y_train_clean,
    exogenous=X_train_clean,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    n_jobs=-1,
)
arimax_fit = arimax.fit(y_train_clean, X_train_clean)

joblib.dump(arimax_fit, ARTM / "arimax.pkl")

fc_arimax = arimax_fit.predict_in_sample(X=X_train_clean)
fcast_arimax = arimax_fit.predict(
    n_periods=len(y_test_clean),
    X=X_test_clean
)
pd.Series(fcast_arimax, index=y_test_clean.index).to_csv(ARTF / "arimax_test.csv")


res_arimax = y_train_clean - fc_arimax 
res_arimax.to_csv(ARTR / "arimax_residuals.csv")

print("ARIMAX done:", arimax_fit.order)
