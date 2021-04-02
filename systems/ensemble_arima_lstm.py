from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_processing.data_processor import DataProcessor
from models.arima.util import load_arima_parameters
from models.lstm.training_util import load_lstm_model
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, settings) -> Tuple[np.ndarray, dict]:
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

        # LSTM Prediction
        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()
        lstm_model = settings["lstm_models"][ticker]
        lstm_prediction = lstm_model.predict_last(predictors=predictors)

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
        combined_prediction = lstm_prediction + arima_prediction
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

    # Load ARIMA parameters
    settings["arima_params"] = {ticker: load_arima_parameters(ticker) for ticker in futures_list}

    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
