from dataclasses import dataclass


@dataclass
class MomentumConfig:
    lookback: int = 252
    skip: int = 21
    quantile: float = 0.2
    rebalance: int = 21
