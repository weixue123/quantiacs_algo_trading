import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from models.lstm.lstm_model import LSTMModel
from utils.data_processor import build_predictors_and_labels
from systems.settings import get_settings


def myTradingSystem(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                    VOL: np.ndarray, settings) -> Tuple[List[int], dict]:
    print(f"Date: {DATE[-1]}")
    positions: List[float] = []
    dates: List[pd.datetime] = list(map(lambda date_int: pd.to_datetime(date_int, format="%Y%m%d"), DATE))

    for index, asset in enumerate(settings["markets"]):
        if asset == "CASH":
            positions.append(0)

        else:
            ohclv_data = pd.DataFrame(index=dates)
            ohclv_data["open"] = OPEN[:, index]
            ohclv_data["high"] = HIGH[:, index]
            ohclv_data["low"] = LOW[:, index]
            ohclv_data["close"] = CLOSE[:, index]
            ohclv_data["volume"] = VOL[:, index]
            predictors, labels = build_predictors_and_labels(ohclv_data=ohclv_data)

            model = settings["models"][asset]

            if not model.is_trained():
                print(f"Training model for: {asset}")
                model.build_and_train_model(predictors=predictors, labels=labels)

            print(f"Predicting for: {asset}")
            prediction = model.predict_last(predictors=predictors)
            positions.append(1 if prediction == 1 else 0)

    total_weights = np.nansum(np.abs(positions))
    if total_weights == 0:
        # Re-adjust to avoid division by zero
        positions[0] = 1
        total_weights = 1

    weights = np.array(positions) / total_weights

    return weights, settings


def mySettings():
    settings = get_settings()

    # Overwrites the list of futures
    # Randomly choose 10 assets because backtesting on all 88 takes too long
    futures_list = settings["markets"]
    settings["markets"] = ["CASH", *random.sample(futures_list, 10)]

    settings["models"] = {asset: LSTMModel(time_step=5) for asset in futures_list}
    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
