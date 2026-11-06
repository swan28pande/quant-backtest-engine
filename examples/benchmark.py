"""Throughput of the C++ target-weight loop vs. an equivalent pure-Python loop."""
import time

import numpy as np

import qbt

prices = qbt.data.synthetic_ohlcv(n_days=2500, n_assets=200, seed=2)
w = np.sign(np.random.default_rng(0).normal(size=prices["close"].shape)) / 200
cfg = dict(commission_bps=0.5, half_spread_bps=1.0, impact_coef=0.0, max_participation=1.0)

bt = qbt.Backtest(prices, **cfg)
import pandas as pd
wdf = pd.DataFrame(w, prices["close"].index, prices["close"].columns)
t0 = time.perf_counter()
res = bt.run(weights=wdf)
cpp = time.perf_counter() - t0


def python_loop(o, c, w, cash=1e6, comm=0.5e-4, hs=1e-4):
    T, N = c.shape
    pos, pending, eq = np.zeros(N), np.zeros(N), np.empty(T)
    for t in range(T):
        for s in range(N):                      # fills at open
            if pending[s]:
                px = o[t, s] * (1 + np.sign(pending[s]) * hs)
                cash -= pending[s] * px + abs(pending[s]) * px * comm
                pos[s] += pending[s]
                pending[s] = 0
        eq[t] = cash + (pos * c[t]).sum()
        if t + 1 < T:
            for s in range(N):                  # rebalance at close
                pending[s] = w[t, s] * eq[t] / c[t, s] - pos[s]
    return eq


t0 = time.perf_counter()
eq_py = python_loop(prices["open"].to_numpy(), prices["close"].to_numpy(), w)
py = time.perf_counter() - t0
T, N = w.shape
print(f"{T} bars x {N} assets, {len(res.fills):,} fills")
print(f"C++ core : {cpp * 1e3:8.1f} ms  ({T * N / cpp / 1e6:.1f}M bar-assets/s)")
print(f"Python   : {py * 1e3:8.1f} ms  -> {py / cpp:.0f}x slower")
print(f"max equity diff vs reference: {np.abs(res.equity.to_numpy() - eq_py).max():.6f}")
