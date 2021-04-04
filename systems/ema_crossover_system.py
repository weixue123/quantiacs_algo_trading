from typing import List, Tuple

import numpy as np

from data_processing.data_processor import DataProcessor
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe


def myTradingSystem(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                    VOL: np.ndarray, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that uses the simple moving average crossover strategy to predict future trends.
    """

    print(f"Date: {DATE[-1]}")

    positions: List[int] = []

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        fast_periods = 12
        slow_periods = 26

        print(f"Predicting for: {ticker}")
        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = (DataProcessor(data=ohclv_data)
                      .add_ema(periods=fast_periods)
                      .add_ema(periods=slow_periods)
                      .get_data())

        fast_ema = predictors[f"{fast_periods}-PERIOD EMA"].iloc[-1]
        slow_ema = predictors[f"{slow_periods}-PERIOD EMA"].iloc[-1]
        fast_ema_is_increasing = ((predictors[f"{fast_periods}-PERIOD EMA"].iloc[-1]
                                   - predictors[f"{fast_periods}-PERIOD EMA"].iloc[-2])) > 0

        # Fast EMA crosses above slow SMA and fast EMA appears to be increasing - buy signal
        if fast_ema > slow_ema and fast_ema_is_increasing:
            positions.append(1)

        # Fast EMA crosses below slow SMA and fast EMA appears to be decreasing - sell signal
        elif slow_ema > fast_ema and not fast_ema_is_increasing:
            positions.append(-1)

        else:
            positions.append(0)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag=2)
    settings["markets"] = ["CASH", *futures_list]
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
