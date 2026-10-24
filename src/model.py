"""Simple forecasting models and feature preparation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

def country_series(df, country=None):
    work = df if country is None else df[df["country"].str.casefold() == country.casefold()]
    if work.empty:
        raise ValueError(f"No data for country: {country}")
    return work.groupby("date", as_index=False)["revenue"].sum().sort_values("date")

def make_features(series):
    """Use recent lags to predict next-day revenue."""
    values = series["revenue"].astype(float).to_numpy()
    if len(values) < 15:
        raise ValueError("Need at least 15 daily observations")
    X, y = [], []
    for i in range(7, len(values)):
        X.append([values[i-1], values[i-2], values[i-7], float(i % 7)])
        y.append(values[i])
    return np.asarray(X), np.asarray(y)

def train_and_compare(df):
    """Compare last-value baseline with a Random Forest using a time-ordered holdout."""
    total = df.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    X, y = make_features(total)
    split = max(int(len(y) * 0.8), 1)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    if len(y_test) == 0:
        raise ValueError("Not enough data for a holdout test")
    baseline_pred = X_test[:, 0]
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    metrics = {
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
        "random_forest_mae": float(mean_absolute_error(y_test, model_pred)),
        "holdout_rows": int(len(y_test))
    }
    return model, metrics, (y_test, baseline_pred, model_pred)

def forecast_next_30(df, country=None):
    """Produce a 30-day iterative forecast using a fitted Random Forest."""
    series = country_series(df, country)
    values = list(series["revenue"].astype(float))
    if len(values) < 15:
        raise ValueError("Need at least 15 daily observations")
    X, y = make_features(series)
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(X, y)
    forecasts = []
    for i in range(30):
        pos = len(values)
        row = [[values[-1], values[-2], values[-7], float(pos % 7)]]
        pred = max(0.0, float(model.predict(row)[0]))
        forecasts.append(pred)
        values.append(pred)
    return forecasts
