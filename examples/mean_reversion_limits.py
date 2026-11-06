"""Event-driven strategy using limit orders: buy short-term losers below the close, exit after N bars."""
import numpy as np

import qbt

prices = qbt.data.synthetic_ohlcv(n_days=750, n_assets=20, seed=1)
close = prices["close"].to_numpy()
ret5 = np.full_like(close, np.nan)
ret5[5:] = close[5:] / close[:-5] - 1
HOLD, BUDGET = 5, 0.05
entered = {}


def on_bar(ctx: qbt.Engine, t: int):
    # exits: flatten positions held for HOLD bars with a market order
    for sym, t_in in list(entered.items()):
        if t - t_in >= HOLD and ctx.position(sym) != 0:
            ctx.submit(sym, -ctx.position(sym))
            del entered[sym]
    # entries: 3 worst 5-day performers, resting bid 0.5% below the close, valid for 1 bar
    if np.isnan(ret5[t]).all():
        return
    for sym in np.argsort(ret5[t])[:3]:
        if sym in entered or ctx.position(sym) != 0 or ctx.pending_qty(sym) != 0:
            continue
        px = close[t, sym] * 0.995
        ctx.submit(int(sym), BUDGET * ctx.equity / px, limit=px, tif_bars=1)
        entered[int(sym)] = t


res = qbt.Backtest(prices, half_spread_bps=2.0).run(strategy=on_bar)
f = res.fills
print(f"{res.events:,} events, {len(f):,} fills ({(f.qty > 0).sum()} buys, {(f.qty < 0).sum()} sells)")
print(res.stats().round(3).to_string())
