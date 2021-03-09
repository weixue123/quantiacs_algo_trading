import numpy as np
import pandas as pd

__all__ = ["daily_returns", "spread", "volatility", "mac", "rsi"]


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def spread(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low


def volatility(close: pd.Series, lookback: int) -> pd.Series:
    vol = close.rolling(lookback).std(ddof=1)
    return vol * np.sqrt(252)


def mac(close: pd.Series, slow_periods: int, fast_periods: int) -> pd.Series:
    data = pd.DataFrame(index=close.index)

    data["slow_sma"] = close.rolling(slow_periods).mean()
    data["fast_sma"] = close.rolling(fast_periods).mean()
    data["fast_sma_delta"] = data["fast_sma"].pct_change()

    # Returns 1 for golden cross, -1 for death cross, and 0 otherwise
    def crossover(slow_sma: float, fast_sma: float, fast_sma_delta: float) -> int:
        if fast_sma > slow_sma and fast_sma_delta > 0:
            return 1
        elif fast_sma < slow_sma and fast_sma_delta < 0:
            return -1
        else:
            return 0

    return data.apply(lambda row: crossover(row["slow_sma"], row["fast_sma"], row["fast_sma_delta"]), axis=1)


def rsi(close: pd.Series, lookback: int):
    returns = close.pct_change()

    assert lookback <= len(returns), f"Not enough rows to calculate RSI with lookback of {lookback} periods"

    # Set initial average gains and losses
    first_date = returns.index[lookback]
    initial_returns = returns[:first_date]
    initial_average_gain = initial_returns[initial_returns >= 0].mean()
    initial_average_loss = -initial_returns[initial_returns < 0].mean()
    average_gains = pd.Series(data=[initial_average_gain], index=[first_date], dtype=float)
    average_losses = pd.Series(data=[initial_average_loss], index=[first_date], dtype=float)

    # Set subsequent average gains and losses
    for date in returns.index[lookback + 1:]:
        current_returns = returns[date]

        average_gains[date] = (average_gains[-1] * (lookback - 1) + max(current_returns, 0)) / lookback
        average_losses[date] = (average_losses[-1] * (lookback - 1) - min(current_returns, 0)) / lookback

    # Compute RSI
    result = pd.Series(index=average_gains.index, dtype=float)
    for date in result.index:
        try:
            ratio = average_gains[date] / average_losses[date]
            result[date] = 100 - 100 / (1 + ratio)
        except ZeroDivisionError:
            result[date] = 0

    return result
