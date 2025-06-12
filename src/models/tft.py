"""
Temporal Fusion Transformer training script
Artifacts:
  artifacts/models/tft.pth
  artifacts/forecasts/tft_test.csv
  artifacts/metrics/advanced_dl_metrics.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# ── constants ────────────────────────────────────────────────────────────
MAX_EPOCHS = 20
BATCH_SIZE = 64
ENC_LEN    = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "processed" / "nasdaq_features.parquet"
ARTM   = ROOT / "artifacts" / "models"
ARTF   = ROOT / "artifacts" / "forecasts"
METR   = ROOT / "artifacts" / "metrics"
for p in (ARTM, ARTF, METR):
    p.mkdir(parents=True, exist_ok=True)

# ── dataset ──────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA).dropna().reset_index()
df["time_idx"] = np.arange(len(df))
df["group"] = 0

cutoff = int(len(df) * 0.9)
train_df = df.iloc[:cutoff]
val_df   = df.iloc[cutoff:]

cols = list(df.columns)
cols.remove("log_ret")
cols.remove("time_idx")
cols.remove("group")

train_ds = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="log_ret",
    group_ids=["group"],
    time_varying_unknown_reals=["log_ret"] + cols,
    max_encoder_length=ENC_LEN,
    max_prediction_length=1,
)
val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df)

train_dl = train_ds.to_dataloader(batch_size=BATCH_SIZE, num_workers=0)
val_dl   = val_ds.to_dataloader(batch_size=BATCH_SIZE, num_workers=0)

# ── model ────────────────────────────────────────────────────────────────
model = TemporalFusionTransformer.from_dataset(
    train_ds,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
)
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
    enable_progress_bar=False,
)

trainer.fit(model, train_dl, val_dl)
model.save(ARTM / "tft.pth")

# ── forecast & metrics ───────────────────────────────────────────────────
preds = trainer.predict(model, dataloaders=val_dl)
preds = torch.cat(preds).squeeze().cpu().numpy()
truth = val_df["log_ret"].values[ENC_LEN:]
idx   = val_df["Date"].iloc[ENC_LEN:]

pd.Series(preds, index=idx).to_csv(ARTF/"tft_test.csv")

mae  = MeanAbsoluteError()(torch.tensor(preds), torch.tensor(truth)).item()
rmse = MeanSquaredError(squared=False)(torch.tensor(preds), torch.tensor(truth)).item()

eps = 1e-8
safe_truth = np.where(np.abs(truth) < eps, eps, truth)
mape = np.mean(np.abs((truth - preds) / safe_truth)) * 100

pd.DataFrame({"model":["TFT"],"MAE":[mae],"RMSE":[rmse],"MAPE":[mape]}).to_csv(
    METR/"advanced_dl_metrics.csv", index=False)

print("Saved model →", ARTM/"tft.pth")
print("Saved forecast →", ARTF/"tft_test.csv")

