from typing import List, Tuple

import numpy as np
import pandas as pd

from models.lstm.lstm_model import LSTMModel
from utils.data_processor import build_labels, build_predictors
from utils.futures_list import futures_list


def myTradingSystem(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                    VOL: np.ndarray, settings) -> Tuple[List[int], dict]:
    print(f"Date: {DATE[-1]}")
    positions: List[float] = []
    dates: List[pd.datetime] = list(map(lambda date_int: pd.to_datetime(date_int, format="%Y%m%d"), DATE))

    for index, asset in enumerate(settings["markets"]):
        if asset == "CASH":
            positions.append(0)

        else:
            model = settings["models"][asset]

            ohclv_data = pd.DataFrame(index=dates)
            ohclv_data["open"] = OPEN[:, index]
            ohclv_data["high"] = HIGH[:, index]
            ohclv_data["low"] = LOW[:, index]
            ohclv_data["close"] = CLOSE[:, index]
            ohclv_data["volume"] = VOL[:, index]
            predictors = build_predictors(ohclv_data=ohclv_data)
            labels = build_labels(ohclv_data=ohclv_data)

            if not model.is_trained():
                print(f"Training model for: {asset}")
                model.build_and_train_model(predictors=predictors, labels=labels)

            print(f"Predicting for: {asset}")

            prediction = model.predict_last(predictors=predictors)
            positions.append(1 if prediction == 1 else 0)

    total_weights = np.nansum(np.abs(positions))
    if total_weights == 0:
        weights = np.array(positions)
        weights[0] = 1
    else:
        weights = np.array(positions) / total_weights

    return weights, settings


def mySettings():
    settings = {'markets': ['CASH', 'F_ES'],
                'lookback': 504,
                'budget': 10 ** 6,
                'slippage': 0.05,
                'beginInSample': '20180101',
                'endInSample': '20201231',
                'models': {asset: LSTMModel(time_step=5) for asset in futures_list}}

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
