import os
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from data_processing.data_processor import DataProcessor
from systems.systems_util import get_futures_list, get_settings, build_ohclv_dataframe, normalize_weights


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, settings):
    current_date = pd.to_datetime(DATE[-1], format="%Y%m%d")
    positions = []

    print(f"Testing: {current_date.strftime('%Y-%m-%d')}")

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        print(f"Predicting for: {ticker}")

        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()
        predictors_last = predictors.loc[current_date:]

        # On some days, data might not be available
        if len(predictors_last) == 0:
            positions.append(0)

        xgb_model = settings["xgb_models"][ticker]
        xgb_prediction = int(xgb_model.predict(predictors_last.to_numpy())[0])
        positions.append(xgb_prediction)

    weights = normalize_weights(positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]
    settings["xgb_models"] = {ticker: load_xgb_model(ticker) for ticker in futures_list}
    return settings


def load_xgb_model(ticker: str) -> XGBClassifier:
    """
    Helper function to load a trained model previously saved as a pickle.
    """
    print(f"Loading XGBoost Model for {ticker}")
    storage_dir = (
            Path(os.path.dirname(__file__)).parent / "models/xgboost/trained_xgboost_models"
    )
    pickle_in = open(f"{storage_dir}/{ticker}.pkl", "rb")
    return pickle.load(pickle_in)


if __name__ == "__main__":
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
