import json
import os
from pathlib import Path

import pmdarima

from systems.systems_util import get_futures_list
from utils.data_loader import load_processed_data


def build_arima():
    futures_list = get_futures_list(filter_insignificant_lag=1)
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

        params = {"p": p, "q": q, "d": d, "residuals": list(arima_residuals)}

        # Save model
        storage_dir = Path(os.path.dirname(__file__)) / "params"
        os.makedirs(os.path.dirname(storage_dir), exist_ok=True)
        with open(f'{storage_dir}/{ticker}_params.txt', 'w') as f:
            json.dump(params, f, ensure_ascii=False)
            print(f"Saved parameters for {ticker}")


if __name__ == "main":
    build_arima()
