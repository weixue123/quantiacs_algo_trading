from typing import Tuple, List

import numpy as np


from data_processing.data_processor import DataProcessor
from systems.systems_util import get_futures_list, get_settings, normalize_weights, build_ohclv_dataframe
from training_script import train

def myTradingSystem(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                    VOL: np.ndarray, settings) -> Tuple[np.ndarray, dict]:
    """
    Base trading system that simply longs every asset with equal weight.
    """
    print(f"Date: {DATE[-1]}")

    positions: List[int] = []

    for index, ticker in enumerate(settings["markets"]):
        if ticker == "CASH":
            positions.append(0)
            continue

        ohclv_data = build_ohclv_dataframe(DATE, OPEN, HIGH, LOW, CLOSE, VOL, index)
        predictors = DataProcessor(data=ohclv_data).build_predictors()
        prediction = train(ticker).predict(predictors.to_numpy())
        predicted_labels = {0 : -1, 
                            1 : 1,
                            2: 0}
        predictions = [predicted_labels[np.argmax(x)] for x in prediction]

        predicted_labels = {0 : -1, 
                            1 : 1,
                            2: 0}

        print(predictions)
        try:
            positions.append(predictions[-1])
        except:
            positions.append(0)

    weights = normalize_weights(weights=positions)

    return weights, settings


def mySettings():
    settings = get_settings()
    settings["markets"] = ["CASH", *get_futures_list(filter_insignificant_lag_1_acf=True)]
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)