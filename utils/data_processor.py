from typing import Tuple

import pandas as pd

from utils.indicators import *

__all__ = ["build_predictors_and_labels"]


def build_predictors_and_labels(ohclv_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    For a particular asset, prepare the predictors and labels to be input into an ML model.

    The output labels array will be a 1D array containing two values:
        - 1 indicating that the price of the asset rose the next day.
        - 0 indicating that the price of the asset fell the next day.

    :param ohclv_data:
        A dataframe with the daily open, high, low, close and volume data of the specified asset.

    :return:
        A DataFrame of predictors and a Series of labels.
    """
    ohclv_data = ohclv_data.rename(columns=str.lower)

    predictors = _build_indicators_dataframe(ohclv_data=ohclv_data)

    daily_returns_lagged = daily_returns(price=ohclv_data["close"]).shift(-1).dropna()
    labels = daily_returns_lagged.map(lambda returns: 1 if returns >= 0 else 0)

    return predictors, labels


def _build_indicators_dataframe(ohclv_data: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(ohclv_data.index, pd.DatetimeIndex), "Input OHCLV dataframe does not have a datetime index"

    close = ohclv_data["close"]
    high = ohclv_data["high"]
    low = ohclv_data["low"]
    volume = ohclv_data["volume"]

    indicators = pd.DataFrame(index=ohclv_data.index)

    indicators["DAILY RETURNS"] = daily_returns(price=close)
    indicators["SPREAD"] = daily_high_low_spread(high=high, low=low)
    indicators["VOLUME"] = volume
    indicators["VOLATILITY"] = annualized_rolling_volatility(price=close, lookback=21)
    indicators["MAC"] = moving_average_crossover(price=close, slow_periods=50, fast_periods=15)
    indicators["RSI"] = relative_strength_index(price=close, lookback=14)

    return indicators.dropna()
