"""Minimal JSON-lines request logging."""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

def write_log(message, log_path):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **message}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
