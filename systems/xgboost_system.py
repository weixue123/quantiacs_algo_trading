import pandas as pd

from data_processing.data_processor import DataProcessor
from models.xgboost.training_util import load_xgb_model
from systems.systems_util import get_futures_list, get_settings, build_ohclv_dataframe, normalize_weights


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, settings):
    """
    Trading system that uses the XGBoost model to predict changes in price.
    """

    current_date = pd.to_datetime(DATE[-1], format="%Y%m%d")
    positions = []
    weight_for_single_asset = 1 / len(settings["markets"])

    print(f"Testing: {current_date.strftime('%Y-%m-%d')}")

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(1)
            continue

        print(f"Predicting for: {ticker}")

        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()
        predictors_last = predictors.loc[current_date:]

        # On some days, data might not be available
        if len(predictors_last) == 0:
            positions.append(0)
            continue

        xgb_model = settings["xgb_models"][ticker]
        xgb_prediction = int(xgb_model.predict(predictors_last.to_numpy())[0])
        xgb_prediction = max(xgb_prediction, 0)
        positions.append(xgb_prediction * weight_for_single_asset)
        if xgb_prediction != 0:
            positions[0] = positions[0] - weight_for_single_asset

    weights = normalize_weights(positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]
    settings["xgb_models"] = {ticker: load_xgb_model(ticker) for ticker in futures_list}
    return settings


if __name__ == "__main__":
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
