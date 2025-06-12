"""
Temporal Convolutional Network training script
Artifacts:
  artifacts/models/tcn.pth
  artifacts/forecasts/tcn_test.csv
  artifacts/metrics/advanced_dl_metrics.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

# ── hyper-parameters ─────────────────────────────────────────────────────
SEQ_LEN   = 30
BATCH     = 128
EPOCHS    = 20
CHANNELS  = 64
LAYERS    = 4
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM   = ROOT / "artifacts" / "models"
ARTF   = ROOT / "artifacts" / "forecasts"
METR   = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METR):
    p.mkdir(parents=True, exist_ok=True)

# ── dataset ──────────────────────────────────────────────────────────────
y = pd.read_parquet(DATA)["log_ret"].dropna().astype("float32")
train = y.loc[: "2023-06-30"]
val   = y.loc["2023-07-01":"2023-12-31"]

test  = y.loc["2024-01-01":]

class SeqDS(Dataset):
    def __init__(self, series):
        self.x = torch.tensor(series.values, dtype=torch.float32)
    def __len__(self):
        return len(self.x) - SEQ_LEN
    def __getitem__(self, i):
        return self.x[i:i+SEQ_LEN], self.x[i+SEQ_LEN].unsqueeze(0)

train_ds = SeqDS(train)
val_ds   = SeqDS(val)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH)

# ── model ────────────────────────────────────────────────────────────────
class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 1
        dilation = 1
        for _ in range(LAYERS):
            layers.append(nn.Conv1d(channels, CHANNELS, 3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            channels = CHANNELS
            dilation *= 2
        self.conv = nn.Sequential(*layers)
        self.fc   = nn.Linear(CHANNELS, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        return self.fc(out[:, :, -1])

model = TCN().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
mae_metric = MeanAbsoluteError().to(DEVICE)
rmse_metric = MeanSquaredError(squared=False).to(DEVICE)

best_val = np.inf
wait = 0
for epoch in range(EPOCHS):
    model.train(); tr_loss = 0
    for xb,yb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb)
        loss.backward(); opt.step(); tr_loss += loss.item()*len(xb)
    tr_loss /= len(train_ds)

    model.eval(); val_loss = 0
    with torch.no_grad():
        for xb,yb in val_dl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += loss_fn(model(xb), yb).item()*len(xb)
    val_loss /= len(val_ds)
    tqdm.write(f"loss={tr_loss:.4e}  val={val_loss:.4e}")
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), ARTM/"tcn.pth")
    else:
        wait += 1
        if wait >= 3:
            break

# ── test forecast ───────────────────────────────────────────────────────
model.load_state_dict(torch.load(ARTM/"tcn.pth"))
model.eval()

test_ds = SeqDS(test)
test_dl = DataLoader(test_ds, batch_size=BATCH)

preds = []
with torch.no_grad():
    for xb,_ in test_dl:
        xb = xb.to(DEVICE)
        preds.append(model(xb).cpu().squeeze().numpy())
preds = np.concatenate(preds)
idx = test.index[SEQ_LEN:]

pd.Series(preds, index=idx).to_csv(ARTF/"tcn_test.csv")

mae  = mae_metric(torch.tensor(preds), torch.tensor(test.values[SEQ_LEN:])).item()
rmse = rmse_metric(torch.tensor(preds), torch.tensor(test.values[SEQ_LEN:])).item()

eps = 1e-8
true_vals = test.values[SEQ_LEN:]
true_vals = np.where(np.abs(true_vals) < eps, eps, true_vals)
mape = np.mean(np.abs((test.values[SEQ_LEN:] - preds)/true_vals))*100

pd.DataFrame({"model":["TCN"],"MAE":[mae],"RMSE":[rmse],"MAPE":[mape]}).to_csv(
    METR/"advanced_dl_metrics.csv", index=False)

print("Saved model →", ARTM/"tcn.pth")
print("Saved forecast →", ARTF/"tcn_test.csv")

