import pandas as pd
import pytest
from src.model import country_series, make_features, train_and_compare

def sample_data():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.DataFrame({"date": list(dates)*2, "country": ["India"]*50+["Canada"]*50,
                         "revenue": [100+i for i in range(50)] + [50+i*2 for i in range(50)]})

def test_country_series_filters_country():
    result = country_series(sample_data(), "India")
    assert len(result) == 50
    assert result["revenue"].iloc[0] == 100

def test_model_comparison_returns_metrics():
    model, metrics, _ = train_and_compare(sample_data())
    assert model is not None
    assert metrics["holdout_rows"] > 0
    assert metrics["baseline_mae"] >= 0
    assert metrics["random_forest_mae"] >= 0

def test_short_series_rejected():
    with pytest.raises(ValueError):
        make_features(pd.DataFrame({"revenue": [1, 2, 3]}))
