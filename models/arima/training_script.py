import pmdarima
import os
import pandas as pd
import pickle
import numpy as np
import json
import pmdarima
from datetime import datetime, timedelta

from utils.data_loader import load_processed_data
from systems.systems_util import get_futures_list

def build_arima():
    futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)
    for ticker in futures_list:
        print(f"{futures_list.index(ticker) + 1}/{len(futures_list)}: {ticker}")
        data = load_processed_data(ticker)
        data = data.loc[:"2020-12-31"]
        data = data.iloc[:-1]   
        data = data[['CLOSE']]
        arima_model = pmdarima.auto_arima(data)
        arima_model = arima_model.fit(data)
        p, d, q = arima_model.order
        arima_residuals = arima_model.arima_res_.resid
        
        params = {"p": p, "q": q, "d":d, "residuals": list(arima_residuals)}
        
        # Save model
        dire = f"./models/arima/param2/"
        os.makedirs(os.path.dirname(dire), exist_ok=True)
        with open(f'{dire}/{ticker}_params.txt', 'w') as f:
            json.dump(params, f, ensure_ascii=False)
            print(f"Saved parameters for {ticker}")

    
    