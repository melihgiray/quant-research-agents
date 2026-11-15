import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_research_agents.data import demo_prices
from quant_research_agents.experiment import compute_metrics, run_experiment, strategy_streams
from quant_research_agents.models import ExperimentConfig


def test_metrics_hand_calculated_including_initial_loss():
    returns = pd.Series([-0.1, 0.1])
    metrics = compute_metrics(returns)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["annual_return"] == pytest.approx(0.99**126 - 1)
    assert metrics["sharpe"] == pytest.approx(0)
    assert metrics["annual_volatility"] == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252))


def test_future_price_changes_cannot_change_past_results():
    prices = demo_prices()
    config = ExperimentConfig()
    original = strategy_streams(prices, config)
    modified = prices.copy()
    modified.iloc[700:] *= np.linspace(1, 5, len(prices) - 700)[:, None]
    after = strategy_streams(modified, config)
    for variant in original:
        pd.testing.assert_frame_equal(
            original[variant].loc[: prices.index[699]], after[variant].loc[: prices.index[699]]
        )


def test_costs_reduce_net_returns_and_entry_is_charged():
    prices = demo_prices()
    free = strategy_streams(prices, ExperimentConfig(cost_bps=0))
    paid = strategy_streams(prices, ExperimentConfig(cost_bps=20))
    for variant in free:
        assert (paid[variant].net <= free[variant].net + 1e-12).all()
        assert paid[variant].cost.iloc[0] > 0
        np.testing.assert_allclose(
            paid[variant].net, paid[variant].gross - paid[variant].turnover * 0.002
        )
    assert paid["baseline"].index.equals(paid["volatility_scaled"].index)


def test_result_has_deterministic_identity_and_provenance():
    first = run_experiment(ExperimentConfig())
    second = run_experiment(ExperimentConfig())
    assert first == second
    assert first["manifest"]["synthetic"] is True
    assert first["experiment_id"] != run_experiment(ExperimentConfig(cost_bps=10))["experiment_id"]


@pytest.mark.parametrize("change", ["duplicate", "unordered", "missing", "zero", "short"])
def test_invalid_panels_rejected(change):
    prices = demo_prices()
    if change == "duplicate":
        prices = pd.concat([prices, prices.iloc[-1:]])
    elif change == "unordered":
        prices = prices.iloc[::-1]
    elif change == "missing":
        prices.iloc[0, 0] = np.nan
    elif change == "zero":
        prices.iloc[0, 0] = 0
    else:
        prices = prices.iloc[:100]
    with pytest.raises(ValueError):
        strategy_streams(prices, ExperimentConfig())


def test_configuration_rejects_impossible_windows_and_nan():
    with pytest.raises(ValidationError):
        ExperimentConfig(lookback=42, skip=42)
    with pytest.raises(ValidationError):
        ExperimentConfig(cost_bps=float("nan"))
