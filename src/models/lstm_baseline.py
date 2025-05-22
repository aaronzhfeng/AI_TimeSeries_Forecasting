"""
GPU LSTM one-step-ahead baseline
Artifacts:
  artifacts/models/lstm_baseline.pt
  artifacts/forecasts/lstm_test.csv
  artifacts/metrics/dl_metrics.csv
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
from pathlib import Path

# ── config ───────────────────────────────────────────────────────────────
SEQ_LEN     = 30          # look-back window
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
HIDDEN      = 64
PATIENCE    = 5           # early-stop
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM   = ROOT / "artifacts" / "models"
ARTF   = ROOT / "artifacts" / "forecasts"
METRIC = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METRIC): p.mkdir(parents=True, exist_ok=True)

# ── dataset class ────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, series: pd.Series):
        self.x = torch.tensor(series.values, dtype=torch.float32)
    def __len__(self): return len(self.x) - SEQ_LEN
    def __getitem__(self, idx):
        i = idx
        return (self.x[i : i+SEQ_LEN],           # seq
                self.x[i+SEQ_LEN].unsqueeze(0))  # target

# ── load series & split ──────────────────────────────────────────────────
y = pd.read_parquet(DATA)["log_ret"].dropna().astype("float32")
train = y.loc[: "2023-06-30"]          #  ⬅️ 90 % train
val   = y.loc["2023-07-01":"2023-12-31"]  # ⬅️ 10 % validation
test  = y.loc["2024-01-01":]

train_ds = SeqDS(train)
val_ds   = SeqDS(val)
test_ds  = SeqDS(test)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ── model ────────────────────────────────────────────────────────────────
class LSTMReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm  = nn.LSTM(1, HIDDEN, num_layers=2,
                             batch_first=True, dropout=0.2)
        self.head  = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        # x: [B, seq]
        x = x.unsqueeze(-1)                 # [B, seq, 1]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])     # last hidden

model = LSTMReg().to(DEVICE)
optim = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

mae_metric = MeanAbsoluteError().to(DEVICE)
rmse_metric = MeanSquaredError(squared=False).to(DEVICE)

# ── training loop with tqdm & early stop ────────────────────────────────
best_val = np.inf
wait = 0
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(train_ds)

    # ---- validation epoch ---------------------------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred   = model(xb)
            val_loss += loss_fn(pred, yb).item() * len(xb)
    val_loss /= len(val_ds)

    tqdm.write(f"loss={epoch_loss:.4e}  val={val_loss:.4e}")

    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), ARTM/"lstm_baseline.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            tqdm.write("Early stopping.")
            break

# ── test forecast ───────────────────────────────────────────────────────
model.load_state_dict(torch.load(ARTM/"lstm_baseline.pt"))
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(DEVICE)
        preds.append(model(xb).cpu().squeeze().numpy())
preds = np.concatenate(preds)
pd.Series(preds, index=test.index[SEQ_LEN:]).to_csv(ARTF/"lstm_test.csv")

# metrics
mae = mae_metric(torch.tensor(preds), torch.tensor(test.values[SEQ_LEN:])).item()
rmse = rmse_metric(torch.tensor(preds), torch.tensor(test.values[SEQ_LEN:])).item()
eps  = 1e-8                                 # avoid /0
deno = np.where(np.abs(test.values[SEQ_LEN:]) < eps, eps, test.values[SEQ_LEN:])
mape = np.mean(np.abs((test.values[SEQ_LEN:] - preds) / deno)) * 100
pd.DataFrame({"MAE":[mae],"RMSE":[rmse],"MAPE":[mape]}).to_csv(METRIC/"dl_metrics.csv", index=False)

print("Saved model →", ARTM/"lstm_baseline.pt")
print("Saved forecast →", ARTF/"lstm_test.csv")
print("Metrics:", mae, rmse, mape)
