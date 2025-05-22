"""
GARCH(1,1) – scaled residuals, version-adaptive.

Outputs
-------
artifacts/models/garch11_scaled.pkl
artifacts/forecasts/garch_variance_scaled.csv
"""
from pathlib import Path
import pandas as pd, joblib, arch

ROOT      = Path(__file__).resolve().parents[2]
RES_FILE  = ROOT / "artifacts" / "residuals"  / "arimax_residuals.csv"
MODEL_OUT = ROOT / "artifacts" / "models"     / "garch11_scaled.pkl"
VAR_OUT   = ROOT / "artifacts" / "forecasts"  / "garch_variance_scaled.csv"

# ── load & scale residuals ───────────────────────────────────────────────
res = pd.read_csv(RES_FILE, index_col=0, parse_dates=True).squeeze()
res_scaled = res * 100.0

# ── build model (try numba backend if available) ────────────────────────
kwargs = dict(p=1, q=1, rescale=False)
try:
    model = arch.arch_model(res_scaled, backend="numba", **kwargs)
except TypeError:
    print("[info] 'backend' flag unsupported – using default CPU backend.")
    model = arch.arch_model(res_scaled, **kwargs)

fit = model.fit(
    disp="off",
    options={"maxiter": 2000, "ftol": 1e-8}
)

print("Convergence flag:", fit.convergence_flag)
print("ω, α1, β1:", fit.params["omega"], fit.params["alpha[1]"], fit.params["beta[1]"])

joblib.dump(fit, MODEL_OUT)

# ── forecast & rescale variance ─────────────────────────────────────────
var_fore_scaled = fit.forecast(horizon=1).variance["h.1"]
var_fore = var_fore_scaled / (100.0**2)
var_fore.to_csv(VAR_OUT)

print("Saved model   →", MODEL_OUT.relative_to(ROOT))
print("Saved variance→", VAR_OUT.relative_to(ROOT))

