import json
from src.logging_utils import write_log

def test_write_log_creates_json_line(tmp_path):
    path = tmp_path / "test.jsonl"
    record = write_log({"endpoint": "/predict", "status": 200}, path)
    assert path.exists()
    saved = json.loads(path.read_text().strip())
    assert saved["endpoint"] == "/predict"
    assert saved["status"] == 200
    assert "timestamp" in record
