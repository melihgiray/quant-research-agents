# Quant Research Copilot

**SYNTHETIC DATA — engineering demonstration, not investment evidence**

Execution mode: scripted

Experiment: `5cb925f98e597372fcf9`

Period: 2020-12-22 to 2023-11-01 (747 observations)

## Computed results

| Variant | Annual return | Annual volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| baseline | 3.76% | 7.72% | 0.5170 | -16.83% |
| volatility_scaled | 3.95% | 7.03% | 0.5859 | -13.94% |

## Interpretation

The two variants were executed over the same synthetic dataset and cost assumptions. The table records their computed outcomes. This scripted demonstration does not measure model reasoning quality.

Automated checks verify structured numerical claims and citation IDs. They do not establish that sources entail every sentence of the interpretation.

## Limitations

- Synthetic integration fixture; these numbers are not evidence of investment performance.
- Fixed parameters, not a walk-forward or statistical-significance study.
- Close-to-close execution approximation with one-day target-weight lag.
- Flat basis-point costs on target-weight changes; no drift adjustment, market impact, borrow fees, or funding costs.
- Sharpe uses zero risk-free rate. The volatility proxy is the equally weighted synthetic universe.
- Scripted responses exercise ADK orchestration and real tools; no Gemini request was made.

## Retrieved evidence

- [fixture-limitations] [Synthetic fixture and simplified cost model](local://docs/METHODOLOGY.md)
- [execution] [Execution timing and look-ahead prevention](https://github.com/melihgiray/quant-research-system/blob/1bdbf1d7f12c8173b070b89c9f07dad36b51ef15/quant_system/backtest/engine.py)
- [momentum] [Cross-sectional momentum and formation windows](https://github.com/melihgiray/quant-research-system/blob/1bdbf1d7f12c8173b070b89c9f07dad36b51ef15/quant_system/signals/momentum.py)

## Reproduction manifest

```json
{
  "engine_version": "0.1.0",
  "dataset_id": "synthetic-sectors-v1",
  "data_sha256": "9a170eb9465123bcda877320bd42507809d044bf414d87ac838ebe095b3128be",
  "synthetic": true,
  "config": {
    "lookback": 252,
    "skip": 21,
    "rebalance": 21,
    "cost_bps": 5.0
  },
  "evaluation_start": "2020-12-22",
  "evaluation_end": "2023-11-01",
  "observations": 747
}
```
