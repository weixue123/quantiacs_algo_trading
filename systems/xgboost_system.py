import pickle
import numpy as np
import os
import sys
from pathlib import Path

project_root_path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(project_root_path) + "\..")

from data_processing.data_processor import DataProcessor
from systems.systems_util import (
    build_ohclv_dataframe,
    get_settings,
    get_futures_list,
    normalize_weights,
)


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    # f = open("best_gen.txt", "a")  # debug
    # print(f"Testing: {DATE[-1]}")  # debug
    nMarkets = len(settings["markets"])
    lookback = settings["lookback"]
    positions = np.zeros(nMarkets, dtype=np.float)

    for index, ticker in enumerate(settings["markets"]):
        try:
            ohclv_data = build_ohclv_dataframe(
                DATE, OPEN, HIGH, LOW, CLOSE, VOL, index
            )  # build dataframe
            predictors = DataProcessor(
                data=ohclv_data
            ).build_predictors()  # create inputs
            predictors = predictors.iloc[-1:]  # get data from the last time frame
            model = settings["models"][ticker]  # load pre-trained model
            prediction = model.predict(
                predictors.to_numpy()
            )  # to_numpy for xgboost model
            positions[index] = prediction[0]

        # for NaN data or futures that will not be traded, set position to 0
        except:
            positions[index] = 0

    weights = normalize_weights(weights=positions)
    # print(weights)  # debug
    # f.write(f"\nDATE: {DATE[-1]}: best population: {weights}")  # debug
    # f.flush()  # debug

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]
    settings["models"] = {ticker: load_model(ticker) for ticker in futures_list}
    return settings


def load_model(ticker: str):
    """
    Helper function to load a trained model previously saved as a pickle.
    """
    storage_dir = (
        Path(os.path.dirname(__file__)).parent / "models/xgboost/trained_xgboost_models"
    )
    pickle_in = open(f"{storage_dir}\\{ticker}.pkl", "rb")
    return pickle.load(pickle_in)


if __name__ == "__main__":
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
