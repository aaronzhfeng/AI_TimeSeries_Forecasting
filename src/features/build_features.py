import pandas as pd
import pandas_ta as ta
from pathlib import Path
from scipy import stats

RAW = Path(__file__).resolve().parents[1] / "data" / "interim" / "clean.parquet"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(RAW)

# Daily log return & pct change
df["log_ret"] = np.log(df["Close"]).diff()
df["pct_change"] = df["Close"].pct_change()

# Technical indicators
df["sma_20"] = ta.sma(df["Close"], length=20)
df["rsi_14"] = ta.rsi(df["Close"], length=14)

# Non-dimensional features
price_z = stats.zscore(df["Close"].dropna())
df.loc[df["Close"].notna(), "price_z"] = price_z

# Example ratio: Close / 20-day SMA
df["close_sma_ratio"] = df["Close"] / df["sma_20"]

df.to_parquet(OUT/"nasdaq_features.parquet")
