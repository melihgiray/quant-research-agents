"""Deterministic synthetic prices: integration fixtures, never investment evidence."""

import hashlib
import io

import numpy as np
import pandas as pd

DATASET_ID = "synthetic-sectors-v1"


def demo_prices() -> pd.DataFrame:
    rng = np.random.default_rng(1729)
    n = 1000
    dates = pd.bdate_range("2020-01-02", periods=n)
    regime_vol = np.where((np.arange(n) // 125) % 2, 0.018, 0.006)
    market = rng.normal(0.00015, regime_vol)
    innovations = rng.normal(0, 0.007, (n, 6))
    drift = np.sin(np.arange(n)[:, None] / 130 + np.arange(6)) * 0.0007
    returns = market[:, None] * 0.65 + innovations + drift
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"SYN_{i}" for i in range(1, 7)])


def data_digest(prices: pd.DataFrame) -> str:
    buffer = io.StringIO()
    prices.to_csv(buffer, float_format="%.10f", date_format="%Y-%m-%d")
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()
