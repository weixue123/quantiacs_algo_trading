from typing import Tuple

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

__all__ = ["LSTMModel"]


class LSTMModel:
    """
    This is a wrapper class for Keras' LSTM model with custom methods implemented to suit
    the algorithmic trading project.
    """

    def __init__(self, time_step: int):
        """
        Initializes a Sequential model.

        :param time_step:
            Number of time steps of predictors to lookback in each sample.
        """
        self.model = Sequential()
        self.trained = False
        self.time_step = time_step
        self.scaler = MinMaxScaler()

    def build_and_train_model(self, predictors: pd.DataFrame, labels: pd.Series, epochs: int = 50):
        """
        Builds and trains an LSTM classification model based on the input predictors and labels.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.
        :param labels:
            Series of labels of 0's or 1's, with dates as the index.
        :param epochs:
            Number of epochs when training the LSTM model.
        """

        assert labels.isin([0, 1]).all(), "Series of labels should contain only 1's or 0's."

        predictors, labels = self._preprocess(predictors=predictors, labels=labels)

        self.reset_model()
        self.model.add(LSTM(100, activation="relu", input_shape=(predictors.shape[1], predictors.shape[2])))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(predictors, labels, epochs=epochs, shuffle=False, verbose=0)
        self.trained = True

    def is_trained(self) -> bool:
        return self.trained

    def predict_last(self, predictors: pd.DataFrame) -> int:
        """
        Makes a prediction based on the latest sample in the predictors dataframe.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.

        :return:
            An integer 1 or 0 indicating the prediction - 1 representing a price increase and 0
            representing a price decrease.
        """
        assert self.is_trained(), "The model has not been trained yet."

        predictors = predictors.iloc[-self.time_step:].to_numpy()

        try:
            predictors = self.scaler.transform(predictors)
            sample = predictors.reshape(1, predictors.shape[0], predictors.shape[1])
            prediction = self.model.predict(sample)
            return 1 if prediction[0][0] >= 0.5 else 0
        except ValueError:
            return 0

    def reset_model(self) -> None:
        self.model = Sequential()
        self.trained = False

    def _preprocess(self, predictors: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses input predictors and labels dataframes/series into a 3D array and a 1D array
        respectively, so as to input them into the LSTM model.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.
        :param labels:
            Series of labels of 0's or 1's, with dates as the index.

        :return:
            A 3D array of predictor variables and a 1D array of labels.
        """
        intersection_index = predictors.dropna().index.intersection(labels.dropna().index)
        predictors = predictors.loc[intersection_index].to_numpy()
        labels = labels.loc[intersection_index].to_numpy()

        predictors = self.scaler.fit_transform(predictors)

        predictors, labels = self._reshape_data(predictors=predictors, labels=labels)

        return predictors, labels

    def _reshape_data(self, predictors: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape input predictors array from two dimensional to three dimensional, introducting an
        additional time step dimension to suit the LSTM model.

        The input predictors array's dimensions will represent [samples, features], whereas that of the output
        predictors array will represent [samples, timesteps, featuers].

        :param predictors:
            2D array of predictor variables.
        :param labels:
            1D array of labels.

        :return:
            A 3D array of predictor variables and a 1D array of labels, the latter trimmed accordingly.
        """
        assert len(predictors.shape) == 2, "Input array is not 2-dimensional"
        assert len(predictors) >= self.time_step, \
            f"Not enough rows to introduce an additional dimension of size {self.time_step}"

        reshaped_predictors = None
        for index in range(self.time_step, len(predictors) + 1):
            row = predictors[index - self.time_step: index]
            row = row.reshape(1, self.time_step, row.shape[1])
            reshaped_predictors = row if reshaped_predictors is None \
                else np.concatenate((reshaped_predictors, row), axis=0)

        # Trim the labels to match the predictors array
        reshaped_labels = labels[self.time_step - 1:]

        return reshaped_predictors, reshaped_labels
