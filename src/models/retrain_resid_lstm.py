"""
Retrain residual LSTM with Optuna-selected hyper-parameters
-----------------------------------------------------------
Run after tune_resid_lstm.py so best_resid_lstm.json exists.
"""

import json, gc, numpy as np, pandas as pd, torch, torch.nn as nn, torchmetrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
ARTR   = ROOT / "artifacts" / "residuals"
ARTF   = ROOT / "artifacts" / "forecasts"
ARTM   = ROOT / "artifacts" / "models"
METR   = ROOT / "artifacts" / "metrics"
HPAR   = ROOT / "artifacts" / "hparams" / "best_resid_lstm.json"
for p in (ARTF, ARTM, METR): p.mkdir(parents=True, exist_ok=True)

RES_FILE = ARTR / "arimax_residuals.csv"
FC_FILE  = ARTF / "arimax_test.csv"          # classical forecast

# ── load best hyper-params ───────────────────────────────────────────────
with open(HPAR) as f:
    params = json.load(f)
SEQ_LEN = int(params["seq_len"])
HIDDEN  = int(params["hidden"])
DROPOUT = float(params["dropout"])
LR      = float(params["lr"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 128
EPOCHS = 30
PATIENCE = 5

# ── residual series & DataLoader ----------------------------------------
resid = pd.read_csv(RES_FILE, index_col=0, parse_dates=True).squeeze().astype("float32")
train = resid.loc[: "2023-12-31"]

class SeqDS(Dataset):
    def __init__(self, series, seq_len):
        self.seq_len = seq_len
        self.x = torch.tensor(series.values, dtype=torch.float32)
    def __len__(self): return len(self.x) - self.seq_len
    def __getitem__(self, i):
        return (self.x[i:i+self.seq_len], self.x[i+self.seq_len].unsqueeze(0))

train_ds = SeqDS(train, SEQ_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# ── model ----------------------------------------------------------------
class ResLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, num_layers=2,   # 2 layers allow dropout
                            batch_first=True, dropout=DROPOUT)
        self.fc   = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ResLSTM().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

best_loss = np.inf; wait = 0
for epoch in range(1, EPOCHS+1):
    model.train(); ep_loss = 0
    for xb,yb in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb); loss = loss_fn(pred, yb)
        loss.backward(); opt.step()
        ep_loss += loss.item()*len(xb)
    ep_loss /= len(train_ds)
    tqdm.write(f"loss={ep_loss:.3e}")
    if ep_loss < best_loss:
        best_loss, wait = ep_loss, 0
        torch.save(model.state_dict(), ARTM/"resid_lstm_best.pt")
    else:
        wait += 1
        if wait >= PATIENCE: break

# ── roll-forward residual predictions for 2024 ---------------------------
model.load_state_dict(torch.load(ARTM/"resid_lstm_best.pt", weights_only=True))
model.eval()

classical_idx = pd.read_csv(FC_FILE, index_col=0, parse_dates=True).index
start_date    = classical_idx[0]

# seed window = last SEQ_LEN residuals up to 2023-12-31
seed = resid.loc[: start_date - pd.Timedelta(days=1)].values[-SEQ_LEN:].astype("float32")
pred_series = {}
cur_date = start_date
while cur_date <= classical_idx[-1]:
    inp = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        next_res = model(inp).cpu().item()
    pred_series[cur_date] = next_res
    seed = np.append(seed[1:], next_res)
    idx_pos = classical_idx.get_loc(cur_date)
    if idx_pos + 1 >= len(classical_idx): break
    cur_date = classical_idx[idx_pos + 1]

pred_ser = pd.Series(pred_series)
pred_ser.to_csv(ARTF/"resid_lstm_best.csv")

# ── hybrid forecast & metrics -------------------------------------------
classical_fc = pd.read_csv(FC_FILE, index_col=0, parse_dates=True).squeeze()
hybrid_fc = classical_fc + pred_ser
hybrid_fc.to_csv(ARTF/"hybrid_test_optuna.csv")

truth = pd.read_parquet(ROOT/"data/processed/nasdaq_features.parquet")["log_ret"]
truth_test = truth.loc[hybrid_fc.index]

mae  = torchmetrics.functional.mean_absolute_error(
          torch.tensor(hybrid_fc.values), torch.tensor(truth_test.values)).item()
rmse = torchmetrics.functional.mean_squared_error(
          torch.tensor(hybrid_fc.values), torch.tensor(truth_test.values), squared=False).item()
eps  = 1e-8
deno = np.where(np.abs(truth_test.values)<eps, eps, truth_test.values)
mape = np.mean(np.abs((truth_test.values - hybrid_fc.values)/deno))*100

pd.DataFrame({"MAE":[mae],"RMSE":[rmse],"MAPE":[mape]}).to_csv(
    METR/"hybrid_metrics_optuna.csv", index=False)

print("Saved model          →", ARTM/"resid_lstm_best.pt")
print("Residual forecast    →", ARTF/"resid_lstm_best.csv")
print("Hybrid forecast      →", ARTF/"hybrid_test_optuna.csv")
print("Optuna-tuned metrics:", mae, rmse, mape)
