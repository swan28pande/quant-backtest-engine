"""Synthetic OHLCV data for examples and benchmarks."""
import numpy as np
import pandas as pd


def synthetic_ohlcv(n_days=1000, n_assets=20, seed=0, momentum=0.03):
    """Daily bars from a one-factor model with weak cross-sectional momentum, so strategies have something to find."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    vol = rng.uniform(0.01, 0.025, n_assets)
    beta = rng.uniform(0.6, 1.4, n_assets)
    mkt = rng.normal(0.0003, 0.01, n_days)
    eps = rng.normal(0, 1, (n_days, n_assets)) * vol
    r = np.empty((n_days, n_assets))
    r[0] = beta * mkt[0] + eps[0]
    trail = np.zeros(n_assets)
    for t in range(1, n_days):
        trail = 0.95 * trail + 0.05 * r[t - 1]               # slow-moving trend state
        xs = (trail - trail.mean()) / (trail.std() + 1e-12)
        r[t] = beta * mkt[t] + eps[t] + momentum * vol * xs
    close = 100 * np.exp(np.cumsum(r, 0))
    gap = rng.normal(0, 0.3, (n_days, n_assets)) * vol
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.5, (n_days, n_assets))) * vol)
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.5, (n_days, n_assets))) * vol)
    volume = rng.lognormal(13, 0.4, (n_days, n_assets)).round()
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols)
    return {"open": mk(open_), "high": mk(hi), "low": mk(lo), "close": mk(close), "volume": mk(volume)}
