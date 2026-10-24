import pandas as pd
from src.app import create_app

def make_csv(path):
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": list(dates)*2,
        "country": ["India"]*60 + ["Canada"]*60,
        "revenue": [100+i for i in range(60)] + [70+i*1.5 for i in range(60)]
    })
    df.to_csv(path, index=False)

def test_health_endpoint():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)/"data.csv"; logs = Path(d)/"logs.jsonl"
        make_csv(data)
        client = create_app(data, logs).test_client()
        assert client.get("/health").status_code == 200

def test_country_prediction_and_all_countries():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)/"data.csv"; logs = Path(d)/"logs.jsonl"
        make_csv(data)
        client = create_app(data, logs).test_client()
        one = client.post("/predict?date=2025-01-01&country=India")
        all_c = client.post("/predict?date=2025-01-01")
        assert one.status_code == 200
        assert one.json["country"] == "India"
        assert len(one.json["daily_forecast"]) == 30
        assert all_c.status_code == 200
        assert all_c.json["country"] == "ALL"
        assert logs.exists()

def test_invalid_date_is_rejected():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)/"data.csv"; logs = Path(d)/"logs.jsonl"
        make_csv(data)
        client = create_app(data, logs).test_client()
        assert client.post("/predict?date=tomorrow").status_code == 400
