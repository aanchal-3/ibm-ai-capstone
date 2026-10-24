"""Train and compare models, saving report artifacts."""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data_ingestion import ingest_data, ROOT
from .model import train_and_compare

def main():
    df = ingest_data()
    model, metrics, plots = train_and_compare(df)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "revenue_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    y_test, baseline, predicted = plots
    plt.figure(figsize=(9, 4))
    plt.plot(y_test, label="Actual")
    plt.plot(baseline, label="Baseline (previous day)")
    plt.plot(predicted, label="Random Forest")
    plt.title("Forecast model comparison on time-ordered holdout")
    plt.xlabel("Holdout day")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "model_comparison.png")
    plt.close()
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()
