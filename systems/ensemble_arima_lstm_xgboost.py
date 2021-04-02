from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_processing.data_processor import DataProcessor
from models.arima.util import load_arima_parameters
from models.lstm.training_util import load_lstm_model
from models.xgboost.training_util import load_xgb_model
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that ensembles the ARIMA, LSTM and XGBoost model to predict changes in price.
    """
    current_date: pd.Timestamp = pd.to_datetime(DATE[-1], format="%Y%m%d")
    positions = []

    print(f"Testing: {current_date.strftime('%Y-%m-%d')}")

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        print(f"Predicting for: {ticker}")

        # Data Preparation
        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()

        # LSTM Prediction
        lstm_model = settings["lstm_models"][ticker]
        # On some days, data might not be available
        if len(predictors) >= lstm_model.time_step:
            if predictors.index[-1] == current_date:
                lstm_prediction = lstm_model.predict_last(predictors=predictors)
        else:
            lstm_prediction = 0

        # XGBoost Prediction
        xgb_model = settings["xgb_models"][ticker]
        predictors_last = predictors.loc[current_date:]
        # On some days, data might not be available
        if len(predictors_last) == 0:
            xgb_prediction = 0
        else:
            xgb_prediction = int(xgb_model.predict(predictors_last.to_numpy())[0])

        # ARIMA Prediction
        price_data = CLOSE[:, index]
        volatility = pd.Series(price_data).pct_change().std()
        arima_params = settings["arima_params"][ticker]
        arima_model = ARIMA(price_data, order=arima_params)
        forecast = arima_model.fit().forecast()
        forecasted_returns = ((forecast - price_data[-1]) / price_data[-1])[0]
        if forecasted_returns > 0.1 * volatility:
            arima_prediction = 1
        elif forecasted_returns < -0.1 * volatility:
            arima_prediction = -1
        else:
            arima_prediction = 0

        # Ensemble the predictions
        combined_prediction = lstm_prediction + xgb_prediction + arima_prediction
        if combined_prediction >= 1:
            positions.append(1)
        elif combined_prediction <= -1:
            positions.append(-1)
        else:
            positions.append(0)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()

    # Set futures to trade
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]

    # Load LSTM models
    settings["lstm_models"] = {ticker: load_lstm_model(ticker) for ticker in futures_list}

    # Load XGBoost models
    settings["xgb_models"] = {ticker: load_xgb_model(ticker) for ticker in futures_list}

    # Load ARIMA parameters
    settings["arima_params"] = {ticker: load_arima_parameters(ticker) for ticker in futures_list}

    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
