# Upstream source

Repository: https://github.com/melihgiray/quant-research-system
Revision: 1bdbf1d7f12c8173b070b89c9f07dad36b51ef15
License: MIT; the original notice is preserved in LICENSE.

- momentum.py: quant_system/signals/momentum.py, with only the config import changed.
- switcher.py: quant_system/regime/switcher.py, unchanged.
- config.py: a local compatibility dataclass containing the momentum parameters.

The new experiment runner is intentionally smaller than the original backtest
engine. Its costs are flat basis points per unit of target-weight turnover.
It does not reproduce the original walk-forward, market-impact, or execution model.
