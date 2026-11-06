"""Cross-sectional momentum with target weights (fast path: whole loop in C++)."""
import time

import pandas as pd

import qbt

prices = qbt.data.synthetic_ohlcv(n_days=1500, n_assets=30)
close = prices["close"]

# 3-month momentum skipping the last week; long top quintile / short bottom quintile, dollar neutral
mom = close.shift(5) / close.shift(63) - 1
rank = mom.rank(axis=1, pct=True)
weights = (rank >= 0.8).astype(float) - (rank <= 0.2).astype(float)
weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0.0)
weekly = weights[weights.index.dayofweek == 4].reindex(weights.index).ffill().fillna(0.0)  # rebalance Fridays

for name, cfg in {
    "frictionless": dict(commission_bps=0, half_spread_bps=0, impact_coef=0, max_participation=1.0),
    "realistic": dict(commission_bps=0.5, half_spread_bps=2.0, impact_coef=0.1, max_participation=0.05,
                      min_trade_value=5_000),
}.items():
    bt = qbt.Backtest(prices, **cfg)
    t0 = time.perf_counter()
    res = bt.run(weights=weekly)
    dt = time.perf_counter() - t0
    print(f"\n== {name}  ({res.events:,} events, {len(res.fills):,} fills in {dt * 1e3:.1f} ms)")
    print(res.stats().round(3).to_string())

bench = qbt.analytics.returns(close.mean(axis=1))
print("\nvs equal-weight market:", {k: round(v, 3) for k, v in qbt.analytics.alpha_beta(qbt.analytics.returns(res.equity), bench).items()})
