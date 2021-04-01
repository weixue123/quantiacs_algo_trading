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

        fast_periods = 5
        slow_periods = 15

        print(f"Predicting for: {ticker}")
        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = (DataProcessor(data=ohclv_data)
                      .add_sma(periods=fast_periods)
                      .add_sma(periods=slow_periods)
                      .get_data())

        fast_sma = predictors[f"{fast_periods}-PERIOD SMA"].iloc[-1]
        slow_sma = predictors[f"{slow_periods}-PERIOD SMA"].iloc[-1]
        fast_sma_is_increasing = ((predictors[f"{fast_periods}-PERIOD SMA"].iloc[-1]
                                   - predictors[f"{fast_periods}-PERIOD SMA"].iloc[-2])) > 0

        # Fast SMA crosses above slow SMA and fast SMA appears to be increasing - buy signal
        if fast_sma > slow_sma and fast_sma_is_increasing:
            positions.append(1)

        else:
            positions.append(0)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list()
    settings["markets"] = ["CASH", *futures_list]
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
