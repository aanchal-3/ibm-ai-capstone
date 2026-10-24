"""Flask API for 30-day revenue forecasts."""
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request
from .data_ingestion import ingest_data, ROOT
from .model import forecast_next_30
from .logging_utils import write_log
import time

def create_app(data_path=None, log_path=None):
    app = Flask(__name__)
    data_path = Path(data_path) if data_path else ROOT / "data" / "daily_revenue.csv"
    log_path = Path(log_path) if log_path else ROOT / "logs" / "api.jsonl"

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        started = time.perf_counter()
        date = request.args.get("date")
        country = request.args.get("country")
        if not date:
            return jsonify({"error": "date query parameter is required (YYYY-MM-DD)"}), 400
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "date must use YYYY-MM-DD format"}), 400
        try:
            df = ingest_data(data_path)
            forecasts = forecast_next_30(df, country=country)
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        payload = {
            "date": target_date.isoformat(),
            "country": country or "ALL",
            "forecast_days": 30,
            "forecast_total": round(sum(forecasts), 2),
            "daily_forecast": [round(x, 2) for x in forecasts],
            "response_time_ms": duration_ms
        }
        write_log({"endpoint": "/predict", "country": payload["country"],
                   "response_time_ms": duration_ms, "status": 200}, log_path)
        return jsonify(payload)

    @app.get("/logs")
    def logs():
        if not log_path.exists():
            return jsonify({"logs": []})
        lines = log_path.read_text(encoding="utf-8").splitlines()
        import json
        return jsonify({"logs": [json.loads(line) for line in lines[-100:]]})

    return app
