import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # ← jump two levels:  src/data → src → repo-root
RAW  = ROOT / "data" / "raw"
OUTI = ROOT / "data" / "interim"
OUTI.mkdir(exist_ok=True, parents=True)

def load_prices():
    df = pd.read_csv(RAW/"nasdaq.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def load_macro() -> pd.DataFrame:
    """
    Load macro indicators, tolerate many header variations, and ignore
    non-timestamp columns like 'Country'.
    """
    macro_path = RAW / "economic_indicators.csv"
    df = pd.read_csv(macro_path)

    # --- strip obvious non-time descriptive columns -----------------
    for col in ["Country", "Indicator", "Region"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # --- find the first column that *looks* like a date or a year ----
    time_col = None
    for col in df.columns:
        sample = str(df[col].iloc[0])
        if sample.isdigit() and len(sample) == 4:       # e.g. '2020'
            time_col = col
            df["Date"] = pd.to_datetime(df[col].astype(str) + "-01-01")
            break
        try:
            pd.to_datetime(sample)                      # e.g. '2020-01-31'
            time_col = col
            df["Date"] = pd.to_datetime(df[col])
            break
        except ValueError:
            continue

    if time_col is None:
        raise ValueError("Could not detect a date or year column in macro CSV.")

    df = (
        df.drop(columns=[time_col])        # drop raw time column
          .sort_values("Date")
          .set_index("Date")
    )
    df = df[~df.index.duplicated(keep="first")]   # ensure unique index
    return df



def align_and_fill():
    px, macro = load_prices(), load_macro()

    # master calendar = union of trading days
    idx = pd.date_range(px.index.min(), px.index.max(), freq="B")
    px = px.reindex(idx)
    macro = macro.reindex(idx).ffill()   # forward-fill holidays

    combined = px.join(macro, how="left")
    combined.to_parquet(OUTI/"aligned.parquet")
    print("Aligned dataset saved")

if __name__ == "__main__":
    align_and_fill()
