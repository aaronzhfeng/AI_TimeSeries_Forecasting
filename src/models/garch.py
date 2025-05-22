"""
GARCH(1,1) fit on ARIMAX residuals to model conditional variance.
"""
import pandas as pd, joblib
from arch import arch_model
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "artifacts" / "residuals" / "arimax_residuals.csv"
OUTM = ROOT / "artifacts" / "models" / "garch11.pkl"
OUTF = ROOT / "artifacts" / "forecasts" / "garch_variance.csv"

res = pd.read_csv(RES, index_col=0, parse_dates=True).squeeze()
garch = arch_model(res, p=1, q=1, rescale=False)
fit   = garch.fit(disp="off")
joblib.dump(fit, OUTM)

# 1-step-ahead conditional variance forecasts
var_fore = fit.forecast(horizon=1).variance["h.1"]
var_fore.to_csv(OUTF)

print("GARCH(1,1) fitted.  ω, α1, β1:",
      fit.params["omega"], fit.params["alpha[1]"], fit.params["beta[1]"])

