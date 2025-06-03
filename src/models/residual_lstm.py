"""
Residual-correction LSTM  |  Phase 4 – Hybrid Model
---------------------------------------------------
1.  Train GPU LSTM on past residuals  (t-30 … t-1)  → residual_t
2.  Predict residuals on test set
3.  Hybrid forecast  = classical forecast + predicted residual
4.  Save model, residual forecast, hybrid forecast, and metrics
"""

from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torchmetrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── hyper-params ─────────────────────────────────────────────────────────
SEQ_LEN  = 30
BATCH    = 128
EPOCHS   = 25
LR       = 1e-3
HIDDEN   = 32
PATIENCE = 5
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
ARTR = ROOT / "artifacts" / "residuals"
ARTF = ROOT / "artifacts" / "forecasts"
ARTM = ROOT / "artifacts" / "models"
METR = ROOT / "artifacts" / "metrics"
for p in (ARTF, ARTM, METR): p.mkdir(parents=True, exist_ok=True)

RES_FILE = ARTR / "arimax_residuals.csv"      # best classical residuals
FC_FILE  = ARTF / "arimax_test.csv"           # classical forecast on test

# ── 1.  residual series → train / test split ────────────────────────────
resid = pd.read_csv(RES_FILE, index_col=0, parse_dates=True).squeeze().astype("float32")

train = resid.loc[: "2023-10-31"]        # training residuals
test  = resid.loc["2023-11-01":]         # leaves ≥30 obs for SEQ_LEN
assert len(test) > SEQ_LEN, "Test set too short for chosen SEQ_LEN"

# ── dataset class ────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, series):
        self.x = torch.tensor(series.values, dtype=torch.float32)
    def __len__(self): return len(self.x) - SEQ_LEN
    def __getitem__(self, i):
        return (self.x[i:i+SEQ_LEN], self.x[i+SEQ_LEN].unsqueeze(0))

train_ds = SeqDS(train)
val_split = int(len(train_ds)*0.9)
val_ds   = torch.utils.data.Subset(train_ds, list(range(val_split, len(train_ds))))
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH)
test_ds  = SeqDS(test)
test_dl  = DataLoader(test_ds,  batch_size=BATCH)

# ── 2.  residual-LSTM model ─────────────────────────────────────────────
class ResLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ResLSTM().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
best_val = np.inf; wait = 0

for epoch in range(1, EPOCHS+1):
    model.train(); epoch_loss = 0
    for xb,yb in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()*len(xb)
    epoch_loss /= len(train_ds)
    
    # ---- validation
    model.eval(); val_loss = 0
    with torch.no_grad():
        for xb,yb in val_dl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += loss_fn(model(xb), yb).item()*len(xb)
    val_loss /= len(val_ds)
    tqdm.write(f"loss={epoch_loss:.3e}  val={val_loss:.3e}")
    
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), ARTM/"resid_lstm.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            tqdm.write("Early stop"); break

# # ── 3.  predict residuals on test ────────────────────────────────────────
# model.load_state_dict(torch.load(ARTM/"resid_lstm.pt", weights_only=True))
# model.eval(); preds=[]
# with torch.no_grad():
#     for xb,_ in test_dl:
#         preds.append(model(xb.to(DEVICE)).cpu().squeeze().numpy())
# preds = np.concatenate(preds)
# pred_ser = pd.Series(preds, index=test.index[SEQ_LEN:])
# pred_ser.to_csv(ARTF/"resid_lstm.csv")

# ── 3.  iterative roll-forward residual forecast (covers 2024) ───────────
model.load_state_dict(torch.load(ARTM/"resid_lstm.pt", weights_only=True))
model.eval()

# classical forecast index (2024 trading days)
needed_dates = pd.read_csv(FC_FILE, index_col=0, parse_dates=True).index
start_date   = needed_dates[0]

# seed window = last 30 true residuals up to start_date-1
seed_window = resid.loc[: start_date - pd.Timedelta(days=1)].values[-SEQ_LEN:].astype("float32")

pred_series = {}
cur_date = start_date
while cur_date <= needed_dates[-1]:
    # predict next residual
    seq = torch.tensor(seed_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        next_res = model(seq).cpu().item()
    pred_series[cur_date] = next_res

    # roll window forward
    seed_window = np.append(seed_window[1:], next_res)

    # advance to next trading day in needed_dates
    idx_pos = needed_dates.get_loc(cur_date)
    if idx_pos + 1 >= len(needed_dates):
        break
    cur_date = needed_dates[idx_pos + 1]

pred_ser = pd.Series(pred_series)
pred_ser.to_csv(ARTF / "resid_lstm.csv")

# ── 4.  hybrid combination (align indices) ───────────────────────────────
classical_fc = pd.read_csv(FC_FILE, index_col=0, parse_dates=True).squeeze()

common_idx = pred_ser.index.intersection(classical_fc.index)
if len(common_idx) < len(pred_ser):
    print(f"[info] {len(pred_ser) - len(common_idx)} residual preds "
          "dropped (no matching classical dates).")
pred_ser  = pred_ser.loc[common_idx]
hybrid_fc = classical_fc.loc[common_idx] + pred_ser
hybrid_fc.to_csv(ARTF/"hybrid_test.csv")

# ── 5.  metrics ─────────────────────────────────────────────────────────
truth = pd.read_parquet(ROOT/"data/processed/nasdaq_features.parquet")["log_ret"]
truth_test = truth.loc[common_idx]

mae  = torchmetrics.functional.mean_absolute_error(
          torch.tensor(hybrid_fc.values), torch.tensor(truth_test.values)).item()
rmse = torchmetrics.functional.mean_squared_error(
          torch.tensor(hybrid_fc.values), torch.tensor(truth_test.values), squared=False).item()
eps  = 1e-8
deno = np.where(np.abs(truth_test.values) < eps, eps, truth_test.values)
mape = np.mean(np.abs((truth_test.values - hybrid_fc.values) / deno)) * 100

pd.DataFrame({"MAE":[mae], "RMSE":[rmse], "MAPE":[mape]}).to_csv(
    METR/"hybrid_metrics.csv", index=False)

print("Saved model →", ARTM/"resid_lstm.pt")
print("Residual forecast →", ARTF/"resid_lstm.csv")
print("Hybrid forecast   →", ARTF/"hybrid_test.csv")
print("Hybrid metrics:", mae, rmse, mape)
