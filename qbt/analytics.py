"""Performance statistics for backtest returns."""
import numpy as np
import pandas as pd
from scipy import stats


def returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(r: pd.Series, periods: int = 252) -> float:
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")


def sortino(r: pd.Series, periods: int = 252) -> float:
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(np.sqrt(periods) * r.mean() / downside) if downside > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def cagr(equity: pd.Series, periods: int = 252) -> float:
    years = (len(equity) - 1) / periods
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark), adjusting for skew and kurtosis (Bailey & López de Prado, 2012)."""
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def alpha_beta(r: pd.Series, benchmark: pd.Series, periods: int = 252):
    df = pd.concat([r, benchmark], axis=1, join="inner").dropna()
    fit = stats.linregress(df.iloc[:, 1], df.iloc[:, 0])
    return {"alpha_ann": float(fit.intercept * periods), "beta": float(fit.slope),
            "alpha_t": float(fit.intercept / fit.intercept_stderr)}


def summary(equity: pd.Series, turnover: float = float("nan"), costs: float = float("nan"), periods: int = 252) -> pd.Series:
    r = returns(equity)
    return pd.Series({
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": cagr(equity, periods),
        "vol_ann": r.std(ddof=1) * np.sqrt(periods),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": max_drawdown(equity),
        "psr": probabilistic_sharpe(r),
        "mean_ret_t": stats.ttest_1samp(r, 0.0).statistic,
        "turnover_ann": turnover,
        "costs_pct": costs,
    })
