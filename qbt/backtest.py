"""Pandas-friendly research API over the C++ engine."""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from qbt import analytics
from qbt._core import Config, Engine

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Result:
    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    fills: pd.DataFrame
    events: int
    _close: pd.DataFrame = field(repr=False)
    _open: pd.DataFrame = field(repr=False)

    def stats(self, periods: int = 252) -> pd.Series:
        notional = (self.fills.qty.abs() * self.fills.price).sum() if len(self.fills) else 0.0
        years = max(len(self.equity) / periods, 1e-9)
        turnover = notional / self.equity.mean() / years
        # costs: commission + slippage vs the bar's open (what a frictionless fill would get)
        costs = 0.0
        if len(self.fills):
            f = self.fills
            ref = self._open.to_numpy()[f.t.to_numpy(), f.sym.to_numpy()]
            costs = f.commission.sum() + (f.qty * (f.price - ref)).sum()
        return analytics.summary(self.equity, turnover, costs / self.equity.iloc[0], periods)

    @property
    def weights(self) -> pd.DataFrame:
        return self.positions * self._close / self.equity.values[:, None]


class Backtest:
    """
    prices: dict with wide DataFrames ("open", "high", "low", "close", "volume"), index = dates, columns = symbols.

    Two ways to run:
      bt.run(weights=df)        target weights per date (decided at that date's close) -- fast, pure C++ loop
      bt.run(strategy=fn)       event-driven: fn(ctx: Engine, t: int) called each close; use ctx.submit(...)
    """

    def __init__(self, prices: dict, config: Optional[Config] = None, **config_kwargs):
        close = prices["close"]
        self.index, self.symbols = close.index, list(close.columns)
        self.prices = {k: prices[k].reindex(index=self.index, columns=self.symbols) for k in FIELDS}
        self.config = config or Config(**config_kwargs)

    def _engine(self) -> Engine:
        arr = {k: np.ascontiguousarray(v.to_numpy(np.float64)) for k, v in self.prices.items()}
        return Engine(arr["open"], arr["high"], arr["low"], arr["close"], arr["volume"], self.config)

    def run(self, weights: Optional[pd.DataFrame] = None, strategy: Optional[Callable] = None) -> Result:
        if (weights is None) == (strategy is None):
            raise ValueError("pass exactly one of weights= or strategy=")
        eng = self._engine()
        if weights is not None:
            w = weights.reindex(index=self.index, columns=self.symbols).fillna(0.0)
            eng.run_targets(np.ascontiguousarray(w.to_numpy(np.float64)))
        else:
            eng.run(strategy)
        res = eng.results()
        f = pd.DataFrame(res["fills"])
        if len(f):
            f.insert(0, "date", self.index[f.t.to_numpy()])
            f["symbol"] = np.asarray(self.symbols)[f.sym.to_numpy()]
        return Result(
            equity=pd.Series(res["equity"], self.index, name="equity"),
            cash=pd.Series(res["cash"], self.index, name="cash"),
            positions=pd.DataFrame(res["positions"], self.index, self.symbols),
            fills=f,
            events=res["events"],
            _close=self.prices["close"],
            _open=self.prices["open"],
        )
