import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from data_processing.data_processor import DataProcessor
from models.lstm.lstm_model import LSTMModel
from systems.systems_util import get_settings, normalize_weights, build_ohclv_dataframe


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
    # futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    futures_list = ["F_AD", "F_ES"]
    settings["markets"] = ["CASH", *futures_list]
    settings["models"] = {ticker: load_model(ticker) for ticker in futures_list}
    return settings


def load_model(ticker: str):
    """
    Helper function to load a trained model previously saved as a pickle.
    """
    storage_dir = Path(os.path.dirname(__file__)).parent / "models/lstm/serialized_models"
    pickle_in = open(f"{storage_dir}/{ticker}.pickle", "rb")
    model: LSTMModel = pickle.load(pickle_in)
    assert model.is_trained(), f"Loaded model for future {ticker} is not trained."
    return model


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
