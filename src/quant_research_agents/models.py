from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExperimentConfig(StrictModel):
    lookback: int = Field(default=252, ge=42, le=504)
    skip: int = Field(default=21, ge=1, le=63)
    rebalance: int = Field(default=21, ge=1, le=63)
    cost_bps: float = Field(default=5.0, ge=0, le=100)

    @model_validator(mode="after")
    def ordered_windows(self):
        if self.skip >= self.lookback:
            raise ValueError("skip must be shorter than lookback")
        return self


class Plan(StrictModel):
    decision: Literal["execute", "refuse"]
    reason: str = Field(min_length=1, max_length=2000)
    config: ExperimentConfig
    research_query: str = Field(min_length=1, max_length=500)


class Claim(StrictModel):
    variant: Literal["baseline", "volatility_scaled"]
    metric: Literal["annual_return", "annual_volatility", "sharpe", "max_drawdown"]
    value: float


class Review(StrictModel):
    status: Literal["completed", "refused"]
    summary: str = Field(min_length=1, max_length=4000)
    citations: list[str] = Field(max_length=10)
    claims: list[Claim] = Field(max_length=16)
    limitations: list[str] = Field(min_length=1, max_length=12)
