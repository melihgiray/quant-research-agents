# Experiment methodology

This is a synthetic integration experiment, not a strategy performance study.

## Data

`synthetic-sectors-v1` generates 1,000 business-day observations for six synthetic
assets, starting on 2020-01-02, using NumPy's default random generator with seed
1729. A common shock changes volatility every 125 rows; each asset also has noise
and a smoothly varying drift. Prices are positive exponentiated cumulative returns.
Business days include dates that can be exchange holidays. Names `SYN_1`–`SYN_6`
are not actual instruments. The manifest includes a SHA-256 fingerprint of the
price panel serialized to ten decimal places.

## Signal and exposure

The attributed upstream momentum function ranks `price[t-skip]/price[t-lookback]-1`.
It allocates +0.5 to the winning leg and -0.5 to the losing leg, rebalancing on a
fixed trading-day schedule. Defaults: 252-day lookback, 21-day skip, 21-day rebalance.

The scaled variant halves target exposure when 21-day realized volatility exceeds
1.25 times 252-day volatility, measured on the equal-weight asset-return proxy.
Both variants remain flat through a common warmup. Target weights formed at close
T are shifted once to earn the close-to-close return on row T+1. This assumes
execution at the signal close; it is not a next-open simulation.

## Costs and scoring

Daily cost = sum of absolute changes in held target weights × cost_bps / 10,000.
This intentionally excludes passive weight drift, market impact, borrow costs,
funding and fees. Entry costs are included in the shared post-warmup scoring window.

- Annual return: geometric compound return scaled to 252 observations per year.
- Annual volatility: sample standard deviation times sqrt(252).
- Sharpe: mean / sample standard deviation times sqrt(252), zero risk-free rate.
- Maximum drawdown: minimum equity/running-peak minus one, including initial equity 1.

Metrics are descriptive. No parameter search, walk-forward inference, confidence
interval, or significance claim is made. The finalizer recomputes headline metrics
from stored returns and validates the reviewer’s structured claims against them.
It does not independently certify the trading simulator or the model’s prose.
