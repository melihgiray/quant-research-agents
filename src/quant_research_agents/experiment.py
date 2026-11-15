"""Deterministic computations. LLMs never calculate the performance metrics."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ._vendor.config import MomentumConfig
from ._vendor.momentum import cross_sectional_momentum
from ._vendor.switcher import apply_regime_sizing
from .data import DATASET_ID, data_digest, demo_prices
from .models import ExperimentConfig

ENGINE_VERSION = "0.1.0"


def validate_prices(prices: pd.DataFrame, config: ExperimentConfig) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices need a DatetimeIndex")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    if prices.columns.has_duplicates or prices.shape[1] < 5:
        raise ValueError("at least five distinct assets are required")
    if len(prices) < max(config.lookback, 252) + 22:
        raise ValueError("insufficient history after warmup")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("prices must be finite, positive, and complete")


def compute_metrics(net: pd.Series) -> dict:
    if net.empty or not np.isfinite(net).all() or (net <= -1).any():
        raise ValueError("invalid return stream")
    equity = np.r_[1.0, (1.0 + net).cumprod().to_numpy()]
    vol = float(net.std(ddof=1))
    return {
        "annual_return": float(equity[-1] ** (252 / len(net)) - 1),
        "annual_volatility": float(vol * np.sqrt(252)),
        "sharpe": float(net.mean() / vol * np.sqrt(252)) if vol > 1e-12 else None,
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
    }


def strategy_streams(prices: pd.DataFrame, config: ExperimentConfig) -> dict:
    validate_prices(prices, config)
    weights = cross_sectional_momentum(
        SimpleNamespace(close=prices),
        MomentumConfig(config.lookback, config.skip, 0.2, config.rebalance),
    )
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    # Available at close(T); both target weights and the regime scale are applied
    # at T+1 below. No extra lag is hidden inside this detector.
    proxy = returns.mean(axis=1)
    ratio = proxy.rolling(21).std() / proxy.rolling(252).std().replace(0, np.nan)
    regime = (ratio > 1.25).astype(int)
    scaled = apply_regime_sizing(weights, regime, defensive_scale=0.5)
    # Keep both variants flat until the common warmup ends. First entry costs
    # then fall inside the measured interval for every permitted lookback.
    start = max(config.lookback, 252)
    streams = {}
    for name, target in (("baseline", weights), ("volatility_scaled", scaled)):
        target = target.copy()
        target.iloc[:start] = 0.0
        held = target.shift(1).fillna(0.0)
        gross = (held * returns).sum(axis=1)
        turnover = held.diff().fillna(held).abs().sum(axis=1)
        costs = turnover * config.cost_bps / 10000
        net = gross - costs
        streams[name] = pd.DataFrame(
            {
                "gross": gross,
                "cost": costs,
                "net": net,
                "turnover": turnover,
            }
        ).iloc[start + 1 :]
    return streams


def run_experiment(config: ExperimentConfig, prices: pd.DataFrame | None = None) -> dict:
    fixture = prices is None
    prices = demo_prices() if fixture else prices
    streams = strategy_streams(prices, config)
    manifest = {
        "engine_version": ENGINE_VERSION,
        "dataset_id": DATASET_ID if fixture else "provided-panel",
        "data_sha256": data_digest(prices),
        "synthetic": fixture,
        "config": config.model_dump(),
        "evaluation_start": streams["baseline"].index[0].date().isoformat(),
        "evaluation_end": streams["baseline"].index[-1].date().isoformat(),
        "observations": len(streams["baseline"]),
    }
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    return {
        "experiment_id": identity,
        "manifest": manifest,
        "metrics": {name: compute_metrics(frame.net) for name, frame in streams.items()},
        "series": {
            name: [
                {"date": index.date().isoformat(), **row.to_dict()}
                for index, row in frame.iterrows()
            ]
            for name, frame in streams.items()
        },
        "limitations": [
            "Synthetic integration fixture; these numbers are not evidence of investment performance."
            if fixture
            else "Caller-provided panel; data provenance must be reviewed separately.",
            "Fixed parameters, not a walk-forward or statistical-significance study.",
            "Close-to-close execution approximation with one-day target-weight lag.",
            "Flat basis-point costs on target-weight changes; no drift adjustment, market impact, borrow fees, or funding costs.",
            "Sharpe uses zero risk-free rate. The volatility proxy is the equally weighted synthetic universe.",
        ],
    }
