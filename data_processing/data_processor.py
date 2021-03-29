from typing import Tuple

import pandas as pd

from data_processing.indicators import rolling_volatility, macd, rsi, atr

__all__ = ["DataProcessor"]


class DataProcessor:

    def __init__(self, data: pd.DataFrame):
        """
        Initializes a class for processing a dataframe of an asset's OHCLV data.
        """
        missing = [col for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"] if col not in data.columns]
        assert len(missing) == 0, f"{missing} columns not found in input futures price data."
        self.data = data

    def get_predictors_and_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns a predictor dataframe and a labels series.
        """
        if "LABELS" not in self.data.columns:
            self.add_labels()
        final = self.data.dropna()
        return final.drop("LABELS", axis=1), final["LABELS"]

    def add_labels(self):
        """
        Adds labels, based on the most recent daily returns and volatility, to each row of the dataframe:
            1 if daily returns is positive and greater than 0.5 SD (i.e. volatility)
            0 if daily returns is smaller than 0.5 SD in magnitude
            -1 if daily returns is negative and smaller than 0.5 SD in magnitude
        """

        if "DAILY RETURNS" not in self.data.columns:
            self.add_daily_returns()
        if "ROLLING VOLATILITY" not in self.data.columns:
            self.add_rolling_volatility()
        self.data["DAILY RETURNS NEXT"] = self.data["DAILY RETURNS"].shift(-1)

        labels = []
        for _, row in self.data.iterrows():
            returns_next, vol = row["DAILY RETURNS NEXT"], row["ROLLING VOLATILITY"]
            if returns_next >= 0.5 * vol:
                labels.append(1)
            elif returns_next <= -0.5 * vol:
                labels.append(-1)
            else:
                labels.append(0)

        self.data = self.data.drop("DAILY RETURNS NEXT", axis=1)
        self.data["LABELS"] = labels
        return self

    def add_macd(self, slow_periods: int = 26, fast_periods: int = 12):
        self.data["MACD"] = macd(price=self.data["CLOSE"], slow_periods=slow_periods, fast_periods=fast_periods)
        return self

    def add_rsi(self, lookback: int = 14):
        self.data["RSI"] = rsi(price=self.data["CLOSE"], lookback=lookback)
        return self

    def add_atr(self, periods: int = 14):
        self.data["ATR"] = atr(close=self.data["CLOSE"], high=self.data["HIGH"], low=self.data["LOW"], periods=periods)
        return self

    def add_daily_returns(self):
        self.data["DAILY RETURNS"] = self.data["CLOSE"].pct_change()
        return self

    def add_rolling_volatility(self, lookback: int = 22):
        self.data["ROLLING VOLATILITY"] = rolling_volatility(price=self.data["CLOSE"].pct_change(), lookback=lookback)
        return self

    def add_lag_close(self, lag: int):
        self.data[f"LAG {lag} CLOSE"] = self.data["CLOSE"].shift(lag)
        return self

    def get_data(self):
        return self.data