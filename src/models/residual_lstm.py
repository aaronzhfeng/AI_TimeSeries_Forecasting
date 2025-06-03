"""
Residual-correction LSTM (Hybrid model)

Workflow
--------
1.  Read best classical residuals  → supervised sequences
2.  Train GPU LSTM on residual_t-30…t-1  → residual_t
3.  Predict residuals on 2024 test set
4.  Hybrid forecast = classical forecast + predicted residual
5.  Save:
      artifacts/models/resid_lstm.pt
      artifacts/forecasts/resid_lstm.csv
      artifacts/forecasts/hybrid_test.csv
      artifacts/metrics/hybrid_metrics.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

# ── hyperparams ──────────────────────────────────────────────────────────
SEQ_LEN  = 30
BATCH    = 128
EPOCHS   = 25
LR       = 1e-3
HIDDEN   = 32
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 5

# ── paths ────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
ARTR  = ROOT / "artifacts" / "residuals"
ARTF  = ROOT / "artifacts" / "forecasts"
ARTM  = ROOT / "artifacts" / "models"
METR  = ROOT / "artifacts" / "metrics"
for p in (ARTF, ARTM, METR): p.mkdir(parents=True, exist_ok=True)

RES_FILE = ARTR / "arimax_residuals.csv"   # best classical model
FC_FILE  = ARTF / "arimax_test.csv"        # classical forecast on test

# ── 1. load residual series ──────────────────────────────────────────────
resid = pd.read_csv(RES_FILE, index_col=0, parse_dates=True).squeeze().astype("float32")
train = resid.loc[: "2023-10-31"]          # same split logic
test  = resid.loc["2023-11-01":]           # ensures ≥ 30 rows
assert len(test) > SEQ_LEN, "Test set too short for chosen SEQ_LEN"

# ── dataset & dataloader ─────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, series):
        self.x = torch.tensor(series.values, dtype=torch.float32)
    def __len__(self): return len(self.x) - SEQ_LEN
    def __getitem__(self, i):
        return (self.x[i:i+SEQ_LEN], self.x[i+SEQ_LEN].unsqueeze(0))

train_ds = SeqDS(train)
test_ds  = SeqDS(test)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

# ── 2. model ────────────────────────────────────────────────────────────
class ResLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

model = ResLSTM().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
best_val = np.inf; wait = 0

# simple split for val
val_split = int(len(train_ds)*0.9)
val_ds = torch.utils.data.Subset(train_ds, list(range(val_split, len(train_ds))))
val_dl = DataLoader(val_ds, batch_size=BATCH)

for epoch in range(1, EPOCHS+1):
    model.train(); loss_epoch = 0
    for xb,yb in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
        pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
        loss_epoch += loss.item()*len(xb)
    loss_epoch /= len(train_ds)
    
    # val
    model.eval(); val_loss = 0
    with torch.no_grad():
        for xb,yb in val_dl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += loss_fn(model(xb), yb).item()*len(xb)
    val_loss /= len(val_ds)
    tqdm.write(f"loss={loss_epoch:.3e}  val={val_loss:.3e}")
    
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), ARTM/"resid_lstm.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            tqdm.write("Early stop"); break

# ── 3. predict residuals on test ─────────────────────────────────────────
model.load_state_dict(torch.load(ARTM/"resid_lstm.pt", weights_only=True))
model.eval(); preds=[]
with torch.no_grad():
    for xb,_ in test_dl:
        preds.append(model(xb.to(DEVICE)).cpu().squeeze().numpy())
preds = np.concatenate(preds)
pred_ser = pd.Series(preds, index=test.index[SEQ_LEN:])
pred_ser.to_csv(ARTF/"resid_lstm.csv")

# ── 4. hybrid forecast combine ──────────────────────────────────────────
classical_fc = pd.read_csv(FC_FILE, index_col=0, parse_dates=True).squeeze()
hybrid_fc = classical_fc.loc[pred_ser.index] + pred_ser
hybrid_fc.to_csv(ARTF/"hybrid_test.csv")

# ── 5. metrics ──────────────────────────────────────────────────────────
truth = pd.read_parquet(ROOT/"data/processed/nasdaq_features.parquet")["log_ret"]
truth_test = truth.loc[pred_ser.index]

mae = torchmetrics.functional.mean_absolute_error(
        torch.tensor(hybrid_fc.values), torch.tensor(truth_test.values)).item()
rmse= torchmetrics.functional.mean_squared_error(
        torch.tensor(hybrid_fc.values), torch.tensor(truth_test.values), squared=False).item()
eps = 1e-8
deno= np.where(np.abs(truth_test.values)<eps, eps, truth_test.values)
mape= np.mean(np.abs((truth_test.values - hybrid_fc.values)/deno))*100

pd.DataFrame({"MAE":[mae],"RMSE":[rmse],"MAPE":[mape]}).to_csv(METR/"hybrid_metrics.csv", index=False)

print("Saved model   →", ARTM/"resid_lstm.pt")
print("Saved residual forecast →", ARTF/"resid_lstm.csv")
print("Saved hybrid forecast   →", ARTF/"hybrid_test.csv")
print("Hybrid metrics:", mae, rmse, mape)
