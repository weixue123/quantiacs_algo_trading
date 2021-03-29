from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from data_processing.data_processor import DataProcessor
from models.lstm.lstm_model import LSTMModel
from systems.settings import get_futures_list, get_settings


def myTradingSystem(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                    VOL: np.ndarray, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that uses the LSTM model to predict change in price.
    """

    print(f"Date: {DATE[-1]}")

    positions: List[int] = []

    for index, asset in enumerate(settings["markets"]):
        if asset == "CASH":
            positions.append(0)
            continue

        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        data_processor = DataProcessor(data=ohclv_data)

        model = settings["models"][asset]

        if not model.is_trained():
            print(f"Training model for: {asset}")
            processed_data = data_processor.build_predictors_and_labels()
            predictors, labels = processed_data.drop("LABELS", axis=1), processed_data["LABELS"]
            model.build_and_train_model(predictors=predictors, labels=labels)

        print(f"Predicting for: {asset}")
        predictors = data_processor.build_predictors()
        prediction = model.predict_last(predictors=predictors)
        positions.append(prediction)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", "F_ES"]
    settings["models"] = {asset: LSTMModel(time_step=5) for asset in futures_list}
    return settings


def build_ohclv_dataframe(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                          VOL: np.ndarray, index: int):
    """
    Helper function to build a dataframe of open, high, low, close and volume data.
    """
    dates: List[pd.datetime] = list(map(lambda date_int: pd.to_datetime(date_int, format="%Y%m%d"), DATE))
    ohclv_data = pd.DataFrame(index=dates)
    ohclv_data["OPEN"] = OPEN[:, index]
    ohclv_data["HIGH"] = HIGH[:, index]
    ohclv_data["LOW"] = LOW[:, index]
    ohclv_data["CLOSE"] = CLOSE[:, index]
    ohclv_data["VOL"] = VOL[:, index]
    return ohclv_data


def normalize_weights(weights: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Helper function to normalize an input list or array of weights so that their sum is 1.
    """
    weights = np.array(weights)
    assert len(weights.shape) == 1, "Weights should be a 1-D array."

    total_weights = np.nansum(np.abs(weights))

    # If total weights is zero, adjust the weights for cash (index 0) to 1; this avoids division by zero
    if total_weights == 0:
        weights[0] = 1
        total_weights = 1

    return weights / total_weights


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
