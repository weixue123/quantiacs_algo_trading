import pandas as pd

from utils.indicators import *

__all__ = ["build_predictors", "build_labels"]


def build_predictors(ohclv_data: pd.DataFrame) -> pd.DataFrame:
    """
    For a particular asset, prepare the predictors to be input into an ML model.

    :param ohclv_data:
        A dataframe with the daily open, high, low, close and volume data of the specified asset.

    :return:
        A DataFrame of predictors.
    """
    predictors = _build_indicators_dataframe(ohclv_data=ohclv_data)
    return predictors


def build_labels(ohclv_data: pd.DataFrame) -> pd.Series:
    """
    For a particular asset, prepare the labels of 1's or 0's to be input into an ML model.

    The output labels array will be a 1D array containing two values:
        - 1 indicating that the price of the asset rose the next day.
        - 0 indicating that the price of the asset fell the next day.

    :return:
        A Series of labels.
    """
    daily_returns_lagged = daily_returns(close=ohclv_data["close"]).shift(-1).dropna()
    return daily_returns_lagged.map(lambda returns: 1 if returns >= 0 else 0)


def _build_indicators_dataframe(ohclv_data: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(ohclv_data.index, pd.DatetimeIndex), "Input OHCLV dataframe does not have a datetime index"

    close = ohclv_data["close"]
    high = ohclv_data["high"]
    low = ohclv_data["low"]
    volume = ohclv_data["volume"]

    indicators = pd.DataFrame(index=ohclv_data.index)

    indicators["DAILY RETURNS"] = daily_returns(close=close)
    indicators["SPREAD"] = spread(high=high, low=low)
    indicators["VOLUME"] = volume
    indicators["VOLATILITY"] = volatility(close=close, lookback=21)
    indicators["MAC"] = mac(close=close, slow_periods=50, fast_periods=15)
    indicators["RSI"] = rsi(close=close, lookback=14)

    return indicators.dropna()
