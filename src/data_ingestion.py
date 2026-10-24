"""Data ingestion for the demo capstone. Replace synthetic data with course data when available."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "daily_revenue.csv"
COUNTRIES = ["India", "United Kingdom", "United States", "Canada"]

def make_demo_data(days=420, seed=7):
    """Create reproducible synthetic daily revenue by country."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=days, freq="D")
    rows = []
    for idx, country in enumerate(COUNTRIES):
        base = [900, 1400, 2200, 750][idx]
        for i, date in enumerate(dates):
            weekly = 120 * np.sin(2 * np.pi * i / 7)
            yearly = 90 * np.sin(2 * np.pi * i / 365)
            trend = i * (0.35 + idx * 0.08)
            noise = rng.normal(0, base * 0.08)
            revenue = max(0, base + weekly + yearly + trend + noise)
            rows.append({"date": date, "country": country, "revenue": round(float(revenue), 2)})
    return pd.DataFrame(rows)

def ingest_data(path=DATA_PATH):
    """Load a CSV if present; otherwise generate demo data and save it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        required = {"date", "country", "revenue"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        df = df[list(required)].copy()
    else:
        df = make_demo_data()
        df.to_csv(path, index=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["date", "country", "revenue"]).sort_values(["country", "date"])
    if df.empty:
        raise ValueError("No valid revenue rows found")
    return df.reset_index(drop=True)

if __name__ == "__main__":
    data = ingest_data()
    print(f"Loaded {len(data)} rows across {data['country'].nunique()} countries.")
