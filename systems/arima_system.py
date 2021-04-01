import json
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from systems_util import get_futures_list, get_settings, normalize_weights


def myTradingSystem(DATE, CLOSE, settings) -> Tuple[np.ndarray, dict]:
    print(f"Predicting for: {DATE[-1]}")
    CLOSE = np.transpose(CLOSE)

    positions = []

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        print(f"Predicting for: {ticker}")

        params_dir = f"../models/arima/params/{ticker}_params.txt"
        ticker_returns = CLOSE[index]
        volatility = pd.Series(ticker_returns).pct_change().std()

        with open(params_dir) as f:
            params = json.load(f)
        p = params['p']
        q = params['q']
        d = params['d']
        model = ARIMA(ticker_returns, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast()

        forecasted_returns = ((forecast - ticker_returns[-1]) / ticker_returns[-1])[0]
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
    futures = get_futures_list(filter_insignificant_lag_1_acf=True)
    settings["markets"] = ["CASH", *futures]
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
