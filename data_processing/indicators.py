import numpy as np
import pandas as pd

__all__ = ["rolling_volatility", "sma", "ema", "typical_price", "macd", "rsi", "atr", "cci", "bb", "roc"]


def rolling_volatility(price: pd.Series, lookback: int = 22):
    """
    Given a series of price data, calculates the rolling daily volatility series.
    """
    return price.pct_change().rolling(lookback).std(ddof=1)


def sma(price: pd.Series, periods: int) -> pd.Series:
    """
    Given a series of price data, calculates the simple moving average series.
    """
    return price.rolling(window=periods).apply(lambda s: s.mean())


def ema(price: pd.Series, periods: int) -> pd.Series:
    """
    Given a series of price data, calculates the exponential moving average series.
    """
    # Set alpha to 2 / (N + 1), a commonly used value
    alpha = 2 / (periods + 1)

    # Obtain weights [a(1-a)^(N-1), ..., a(1-a)^2, a(1-a), a] and normalize them so their sum is 1
    weights = [1, *np.cumprod([1 - alpha] * (periods - 1))][::-1]
    weights = alpha * np.array(weights)
    weights = weights / weights.sum()

    return price.rolling(window=periods).apply(lambda s: np.dot(s, weights))


def typical_price(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Given a series of an asset's close, high, and low data, calculates the typical price series.
    """
    return (close + high + low) / 3


def macd(price: pd.Series, slow_periods: int = 26, fast_periods: int = 12) -> pd.Series:
    """
    Give a series of price data, calculates the moving average convergence divergence series.
    """
    slow_ema = ema(price, periods=slow_periods)
    fast_ema = ema(price, periods=fast_periods)
    output = fast_ema - slow_ema

    output.name = "MACD"
    return output


def rsi(price: pd.Series, lookback: int = 14) -> pd.Series:
    """
    Given a series of price data, calculates the relative strength index series.
    """
    returns = price.pct_change()

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
    output = pd.Series(index=average_gains.index, dtype=float)
    for date in output.index:
        try:
            ratio = average_gains[date] / average_losses[date]
            output[date] = 100 - 100 / (1 + ratio)
        except ZeroDivisionError:
            output[date] = 0

    output.name = "RSI"
    return output


def atr(close: pd.Series, high: pd.Series, low: pd.Series, periods: int = 14) -> pd.Series:
    """
    Given the series of an asset's close, high, and low data, calculates the average true range series.
    """
    close.name = "CLOSE"
    high.name = "HIGH"
    low.name = "LOW"
    data = pd.concat([close, high, low], axis=1)

    data["CLOSE PREVIOUS"] = data["CLOSE"].shift(1)
    true_range = pd.concat([data["HIGH"] - data["LOW"],
                            abs(data["HIGH"] - data["CLOSE PREVIOUS"]),
                            abs(data["LOW"] - data["CLOSE PREVIOUS"])
                            ], axis=1).max(axis=1)
    average_true_range = true_range.rolling(periods).mean()

    average_true_range.name = "ATR"
    return average_true_range


def cci(close: pd.Series, high: pd.Series, low: pd.Series, periods: int = 20) -> pd.Series:
    """
    Given a series of an asset's close, high, and low data, calculates the commodity channel index series.
    """
    tp = typical_price(close=close, high=high, low=low)
    moving_average = sma(tp, periods=periods)
    mean_deviation = (tp - moving_average).abs().rolling(window=periods).mean()
    commodity_channel_index = (tp - moving_average) / mean_deviation.multiply(0.015)

    commodity_channel_index.name = "CCI"
    return commodity_channel_index


def bb(close: pd.Series, high: pd.Series, low: pd.Series, upper: bool, periods: int = 20, number_of_sds: float = 2) \
        -> pd.Series:
    """
    Given a series of an asset's close, high, and low data, calculates the bollinger bands series.
    """
    tp = typical_price(close=close, high=high, low=low)
    moving_average = sma(tp, periods=periods)
    moving_sd = tp.rolling(window=periods).std(ddof=1)

    if upper:
        band = moving_average + moving_sd.multiply(number_of_sds)
        band.name = "BB UPPER"
    else:
        band = moving_average - moving_sd.multiply(number_of_sds)
        band.name = "BB LOWER"

    return band


def roc(price: pd.Series, periods: int) -> pd.Series:
    """
    Given a series of price data, calculates the rate of change series.
    """
    return price.pct_change(periods=periods)
