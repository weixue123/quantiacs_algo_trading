import os
from pathlib import Path

import pandas as pd

from data_processing.data_processor import DataProcessor

__all__ = ["load_raw_data", "load_processed_data"]


def load_raw_data(ticker: str):
    """
    Loads raw data from tickerData.
    Processes the raw data into a dataframe with datetime index and the OHCLV data as the columns.
    """
    project_root_path = Path(os.path.dirname(__file__)).parent
    data = pd.read_csv(project_root_path / f"systems/tickerData/{ticker}.txt", delimiter=",")
    data.columns = list(map(lambda col: col.strip(), data.columns))

    data = data.set_index("DATE")
    data.index = pd.to_datetime(data.index, format="%Y%m%d")

    data = data[["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]]
    data = data.sort_index(ascending=True)

    return data


def load_processed_data(ticker: str):
    """
    Loads the predictors and labels dataframe from tickerDataProcessed.

    If the predictors and labels have not been computed before:
        - Load the raw data
        - Build the predictors and labels dataframe
        - Save the processed data into tickerDataProcessed as a CSV file.
    """

    project_root_path = Path(os.path.dirname(__file__)).parent

    try:
        processed_data = pd.read_csv(project_root_path / f"systems/tickerDataProcessed/{ticker}.csv",
                                     delimiter=",",
                                     parse_dates=["DATE"])
        processed_data = processed_data.set_index("DATE")
        return processed_data
    except FileNotFoundError:
        raw_data = load_raw_data(ticker)
        processed_data = DataProcessor(data=raw_data).build_predictors_and_labels()
        processed_data.to_csv(project_root_path / f"systems/tickerDataProcessed/{ticker}.csv")
        processed_data = processed_data.sort_index(ascending=True)
        return processed_data
