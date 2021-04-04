from typing import Tuple

import numpy as np
import pandas as pd

from data_processing.data_processor import DataProcessor
from models.lgbm.training_script import train
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that uses the LightGBM to predict price changes.
    """
    current_date: pd.Timestamp = pd.to_datetime(DATE[-1], format="%Y%m%d")
    positions = []

    print(f"Testing: {current_date.strftime('%Y-%m-%d')}")

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()

        # Model not yet trained
        if ticker not in settings["models"]:
            model = train(ticker)
            settings["models"][ticker] = model

        # Model has been trained in a previous period
        else:
            model = settings["models"][ticker]

        predictors_last = predictors.loc[current_date:]

        # On some days, data might not be available
        if len(predictors_last) == 0:
            positions.append(0)
            continue

        prediction = model.predict(predictors_last)
        predicted_labels = {0: -1, 1: 1, 2: 0}
        prediction = predicted_labels[np.argmax(prediction)]
        positions.append(prediction)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag=2)
    settings["markets"] = ["CASH", *futures_list]
    settings["models"] = {}
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
