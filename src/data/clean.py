import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # ← repo root
INP  = ROOT / "data" / "interim" / "aligned.parquet"
OUT  = ROOT / "data" / "interim"

df = pd.read_parquet(INP)

# Drop rows with no NASDAQ close price (non-trading days remained after reindex)
df = df[df["Close"].notna()]

# Flash-crash spike filter: Winsorise 5-σ   (choose a threshold that suits)
z = (df["Close"] - df["Close"].mean()) / df["Close"].std()
df.loc[z.abs() > 5, "Close"] = np.nan
df["Close"] = df["Close"].interpolate(limit_direction="both")

df.to_parquet(OUT/"clean.parquet")
