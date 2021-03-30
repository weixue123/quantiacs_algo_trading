from typing import List, Tuple

import numpy as np

from data_processing.data_processor import DataProcessor
from models.lstm.training_util import load_model
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe


def myTradingSystem(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                    VOL: np.ndarray, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that uses the LSTM model to predict change in price.
    """

    print(f"Date: {DATE[-1]}")

    positions: List[int] = []

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        print(f"Predicting for: {ticker}")
        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()
        model = settings["models"][ticker]
        prediction = model.predict_last(predictors=predictors)
        positions.append(prediction)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]
    settings["models"] = {ticker: load_model(ticker) for ticker in futures_list}
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
