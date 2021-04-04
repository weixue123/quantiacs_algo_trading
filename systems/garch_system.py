import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from arch import arch_model

from systems_util import get_futures_list, get_settings, normalize_weights


def myTradingSystem(DATE: List[int], CLOSE: np.ndarray, settings) -> Tuple[np.ndarray, dict]:
    
    print(f"Predicting for: {DATE[-1]}")
    CLOSE = np.transpose(CLOSE)[1:]
    log_return = np.diff(np.log(CLOSE))

    positions: List[int] = []

    storage_dir = Path(os.path.dirname(__file__)).parent / "../models/garch/correlation.txt"
    with open(storage_dir) as f:
        cor_dict = json.load(f)

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        print(f"Predicting for: {ticker}")
        
        params_dir = Path(os.path.dirname(__file__)).parent / f"../models/garch/params/{ticker}_params.txt"
        ticker_returns = log_return[:, index-1]
        
        with open(params_dir) as f:
            params = json.load(f)
        
        p = params['p']
        q = params['q']
        fixed_params = params['params']
        model = arch_model(ticker_returns * 10 , p=p, q=q)
        fixed_model = model.fix(fixed_params)
        forecast_vol = fixed_model.forecast()
        var = forecast_vol.variance.iloc[-1:]['h.1']

        # flip the inequality signs lol 
        # if (cor_dict[ticker] > 0.03)
        """
        if (float(np.sqrt(var)) > np.std(ticker_returns)):
            positions.append(1)
        elif (float(np.sqrt(var)) < np.std(ticker_returns)):
            positions.append(-1)
        else:
            positions.append(0)
        """
        if (cor_dict[ticker] < 0.3):
            if (float(np.sqrt(var)) > np.std(ticker_returns)):
                positions.append(1)
            else:
                positions.append(0)
        elif (cor_dict[ticker] > 0.3):
            if (float(np.sqrt(var)) > np.std(ticker_returns)):
                positions.append(-1)
            else:
                positions.append(0)
        else:
            positions.append(0)

    positions = normalize_weights(weights=positions)

    return positions, settings


def mySettings():
    settings = get_settings()
    futures = get_futures_list(filter_insignificant_lag=2)
    futures_list = ["F_AD", "F_ES"]
    settings["markets"] = ["CASH", *futures]
    return settings



if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
