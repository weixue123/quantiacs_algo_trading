from typing import Tuple

import numpy as np
import pandas as pd

from data_processing.data_processor import DataProcessor
from models.lstm.training_util import load_lstm_model
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that uses the LSTM model to predict changes in price.
    """

    current_date: pd.Timestamp = pd.to_datetime(DATE[-1], format="%Y%m%d")
    positions = []

    print(f"Testing: {current_date.strftime('%Y-%m-%d')}")

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(1)
            continue

        print(f"Predicting for: {ticker}")
        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()
        model = settings["models"][ticker]

        # On some days, data might not be available
        if len(predictors) < model.time_step:
            positions.append(0)
        elif predictors.index[-1] != current_date:
            positions.append(0)
        else:
            prediction = model.predict_last(predictors=predictors)
            positions.append(prediction)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]
    settings["models"] = {ticker: load_lstm_model(ticker) for ticker in futures_list}
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
