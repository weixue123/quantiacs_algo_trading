from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

from data_processing.data_processor import DataProcessor
from models.arima.util import load_arima_parameters
from models.lstm.lstm_model import LSTMModel
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

        # LSTM prediction
        lstm_model = settings["lstm_models"][ticker]
        lstm_prediction = get_lstm_prediction(model=lstm_model, predictors=predictors, current_date=current_date)

        # XGBoost prediction
        xgb_model = settings["xgb_models"][ticker]
        xgb_prediction = get_xgb_prediction(model=xgb_model, predictors=predictors, current_date=current_date)

        # ARIMA prediction
        arima_order = settings["arima_params"][ticker]
        arima_prediction = get_arima_prediction(price_data=CLOSE[:, index], order=arima_order)

        # Ensemble the predictions
        combined_prediction = lstm_prediction + xgb_prediction + arima_prediction
        if combined_prediction >= 2:
            positions.append(1)
        elif combined_prediction <= -2:
            positions.append(-1)
        else:
            positions.append(0)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()

    # Set futures to trade
    futures_list = get_futures_list(filter_insignificant_lag=2)
    settings["markets"] = ["CASH", *futures_list]

    # Load LSTM models
    settings["lstm_models"] = {ticker: load_lstm_model(ticker) for ticker in futures_list}

    # Load XGBoost models
    settings["xgb_models"] = {ticker: load_xgb_model(ticker) for ticker in futures_list}

    # Load ARIMA parameters
    settings["arima_params"] = {ticker: load_arima_parameters(ticker) for ticker in futures_list}

    return settings


def get_lstm_prediction(model: LSTMModel, predictors: pd.DataFrame, current_date: pd.Timestamp,
                        allow_short: bool = True) -> int:
    if len(predictors) < model.time_step:
        prediction = 0
    elif predictors.index[-1] != current_date:
        prediction = 0
    else:
        prediction = model.predict_last(predictors=predictors)

    if not allow_short:
        prediction = max(prediction, 0)

    assert prediction in [-1, 0, 1]
    return prediction


def get_xgb_prediction(model: XGBClassifier, predictors: pd.DataFrame, current_date: pd.Timestamp,
                       allow_short: bool = True) -> int:
    predictors_last = predictors.loc[current_date:]
    if len(predictors_last) == 0:
        prediction = 0
    else:
        prediction = int(model.predict(predictors_last.to_numpy())[0])

    if not allow_short:
        prediction = max(prediction, 0)

    assert prediction in [-1, 0, 1]
    return prediction


def get_arima_prediction(price_data: np.ndarray, order=Tuple[int, int, int], allow_short: bool = True) -> int:
    volatility = pd.Series(price_data).pct_change().std(ddof=1)
    model = ARIMA(price_data, order=order)
    forecast = model.fit().forecast(steps=1)
    forecasted_returns = ((forecast - price_data[-1]) / price_data[-1])[0]

    if forecasted_returns > 0.1 * volatility:
        prediction = 1
    elif forecasted_returns < -0.1 * volatility:
        prediction = -1
    else:
        prediction = 0

    if not allow_short:
        prediction = max(prediction, 0)

    assert prediction in [-1, 0, 1]
    return prediction


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
