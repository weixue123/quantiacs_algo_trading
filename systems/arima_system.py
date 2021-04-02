from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from models.arima.util import load_arima_parameters
from systems_util import get_futures_list, get_settings, normalize_weights


def myTradingSystem(DATE, CLOSE, settings) -> Tuple[np.ndarray, dict]:
    """
    Trading system that uses the ARIMA model to predict changes in price.
    """

    current_date: pd.Timestamp = pd.to_datetime(DATE[-1], format="%Y%m%d")
    positions = []

    print(f"Testing: {current_date.strftime('%Y-%m-%d')}")

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        print(f"Predicting for: {ticker}")

        price_data = CLOSE[:, index]
        volatility = pd.Series(price_data).pct_change().std()
        params = settings["params"][ticker]
        model = ARIMA(price_data, order=params)
        model_fit = model.fit()
        forecast = model_fit.forecast()

        forecasted_returns = ((forecast - price_data[-1]) / price_data[-1])[0]

        if forecasted_returns > 0.1 * volatility:
            positions.append(1)
        elif forecasted_returns < -0.1 * volatility:
            positions.append(-1)
        else:
            positions.append(0)

    positions = normalize_weights(weights=positions)
    return positions, settings


def mySettings():
    settings = get_settings()
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures_list]
    settings["params"] = {ticker: load_arima_parameters(ticker) for ticker in futures_list}
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
