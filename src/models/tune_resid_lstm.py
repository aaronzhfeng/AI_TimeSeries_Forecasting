"""
Optuna hyper-search for residual-correction LSTM
------------------------------------------------
Search space
------------
SEQ_LEN   : {14, 21, 30}
HIDDEN    : {16, 32, 64, 96}
DROPOUT   : [0.0, 0.4]
LR        : log-uniform 1e-4 … 3e-3
"""

import json, gc, optuna, torch, torch.nn as nn
import pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError
from tqdm import tqdm
from pathlib import Path

# ── paths & constants ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
ARTR = ROOT / "artifacts" / "residuals"
ARTM = ROOT / "artifacts" / "models"
HPAR = ROOT / "artifacts" / "hparams"
for p in (ARTM, HPAR): p.mkdir(parents=True, exist_ok=True)

RES_FILE = ARTR / "arimax_residuals.csv"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 20
PATIENCE   = 3

# ── dataset class --------------------------------------------------------
class SeqDS(Dataset):
    def __init__(self, series, seq_len):
        self.seq_len = seq_len
        self.x = torch.tensor(series.values, dtype=torch.float32)
    def __len__(self): return len(self.x) - self.seq_len
    def __getitem__(self, i):
        return (self.x[i:i+self.seq_len], self.x[i+self.seq_len].unsqueeze(0))

# ── objective ------------------------------------------------------------
def objective(trial: optuna.trial.Trial):

    # hyper-params
    seq_len = trial.suggest_categorical("seq_len", [14, 21, 30])
    hidden  = trial.suggest_categorical("hidden",  [16, 32, 64, 96])
    drop    = trial.suggest_float("dropout", 0.0, 0.4)
    lr      = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    # load & split residual series
    resid = pd.read_csv(RES_FILE, index_col=0, parse_dates=True).squeeze()
    train = resid.loc[: "2023-10-31"].astype("float32")
    val   = resid.loc["2023-11-01": "2023-12-31"].astype("float32")

    train_ds = SeqDS(train, seq_len); val_ds = SeqDS(val, seq_len)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128)

    # model
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=1,
                                batch_first=True, dropout=drop)
            self.fc   = nn.Linear(hidden, 1)
        def forward(self, xb):
            xb = xb.unsqueeze(-1)
            out, _ = self.lstm(xb)
            return self.fc(out[:, -1, :])

    model = Net().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = np.inf; wait = 0

    for epoch in range(MAX_EPOCHS):
        # ---- train
        model.train(); tr_loss = 0
        for xb,yb in train_dl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb); loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            tr_loss += loss.item()*len(xb)
        # ---- val
        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb,yb in val_dl:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item()*len(xb)
        val_loss /= len(val_ds)

        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), ARTM/"resid_lstm_optuna.pt")
        else:
            wait += 1
            if wait >= PATIENCE: break

    # free GPU memory
    del model; torch.cuda.empty_cache(); gc.collect()
    return best_val

# ── run study ------------------------------------------------------------
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
)
print("Starting Optuna study …")
study.optimize(objective, n_trials=30, timeout=None, show_progress_bar=True)

print("Best params:", study.best_params)
with open(HPAR/"best_resid_lstm.json", "w") as f:
    json.dump(study.best_params, f, indent=2)
print("Saved:", HPAR/"best_resid_lstm.json")
