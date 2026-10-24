"""Create basic EDA plots."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data_ingestion import ingest_data, ROOT

def main():
    df = ingest_data()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    plt.figure(figsize=(10, 4))
    plt.plot(daily["date"], daily["revenue"])
    plt.title("Total daily revenue over time")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(reports / "daily_revenue.png")
    plt.close()
    country = df.groupby("country")["revenue"].sum().sort_values()
    plt.figure(figsize=(8, 4))
    country.plot(kind="bar")
    plt.title("Total revenue by country")
    plt.xlabel("Country")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(reports / "revenue_by_country.png")
    plt.close()
    print(f"EDA charts saved in {reports}")
if __name__ == "__main__":
    main()
