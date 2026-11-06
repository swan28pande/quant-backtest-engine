# Quantitative Research Backtesting Engine

A backtesting and simulation engine. You write research code in Python (NumPy, pandas, SciPy), and a C++17 event-driven core simulates order execution.

**Stack:** C++17 (pybind11), Python, NumPy, pandas, SciPy.

## Design
```
qbt.Backtest (pandas in/out) ──▶ qbt._core.Engine (C++)
                                   event queue per bar: OPEN → INTRABAR → CLOSE
                                   OPEN      orders past their latency activate; market orders fill at the open
                                   INTRABAR  resting limit orders fill if the bar's range strictly crosses the limit
                                   CLOSE     mark-to-market, then the strategy runs (orders → t + latency_bars)
```
Execution model (`qbt.Config`):
- **Commission:** `commission_bps`.
- **Market-order costs:** half-spread (`half_spread_bps`) plus square-root impact (`impact_coef · √(fill / bar volume)`).
- **Partial fills:** each fill is capped at `max_participation` of bar volume, and the rest carries over.
- **No look-ahead:** `latency_bars ≥ 1` is enforced.
- **Limit orders:** time-in-force in bars (`tif_bars`, 0 = good till cancelled).

## Two ways to run
```python
import qbt
prices = qbt.data.synthetic_ohlcv()              # dict of wide DataFrames: open/high/low/close/volume
bt = qbt.Backtest(prices, half_spread_bps=2, impact_coef=0.1, max_participation=0.05)

# 1) Target weights: the whole loop runs in C++ with the GIL released
res = bt.run(weights=weights_df)

# 2) Event-driven: a Python callback at each close, submitting market or limit orders
def on_bar(ctx, t):
    if ctx.position(0) == 0:
        ctx.submit(0, 100, limit=prices["close"].iat[t, 0] * 0.99, tif_bars=1)
res = bt.run(strategy=on_bar)

res.equity, res.positions, res.fills, res.weights
res.stats()   # CAGR, vol, Sharpe, Sortino, max DD, probabilistic Sharpe, t-stat, turnover, costs
```

## Install & examples
```bash
pip install -e .                          # builds the C++ extension (needs a C++17 compiler)
python examples/momentum.py               # cross-sectional momentum, frictionless vs realistic costs
python examples/mean_reversion_limits.py  # event-driven limit-order strategy
python examples/benchmark.py              # C++ core vs pure-Python loop (equity matches exactly)
```
The benchmark runs 2,500 bars × 200 assets with about 500k fills in about 180 ms, roughly 9× faster than the equivalent Python loop.

`costs_pct` in `stats()` is commission plus slippage against the bar's open, as a fraction of starting capital. A negative value means limit orders got price improvement.

## Status
Early skeleton. Next:
- Intraday and tick data.
- Short borrow and financing costs.
- Multiple strategies sharing one account.
- Walk-forward and parameter-sweep helpers.
- Event-level logging.
