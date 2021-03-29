from typing import Tuple

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

__all__ = ["LSTMModel"]


class LSTMModel:
    """
    A wrapper class for Keras' LSTM model with custom methods implemented to suit
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

    def build_and_train_model(self, predictors: pd.DataFrame, labels: pd.Series, epochs: int, hidden_layers: int,
                              verbose=False):
        """
        Builds and trains an LSTM classification model based on the input predictors and labels.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.
        :param labels:
            Series of labels of 0's or 1's, with dates as the index.
        :param epochs:
            Number of epochs when training the LSTM model.
        :param hidden_layers:
            Number of hidden layers
        :param verbose:
            Whether to log training progress.
        """

        assert labels.isin([0, 1, -1]).all(), "Series of labels should contain only 1's, 0's, or -1's."

        predictors, labels = self._preprocess(predictors=predictors, labels=labels)

        self.reset_model()
        self.model.add(LSTM(100, activation="relu", input_shape=(predictors.shape[1], predictors.shape[2])))
        self.model.add(Dropout(0.5))

        for i in range(hidden_layers):
            self.model.add(Dense(100, activation="relu"))
            self.model.add(Dropout(0.5))

        self.model.add(Dense(3, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(predictors, labels, epochs=epochs, shuffle=False, verbose=verbose)
        self.trained = True

    def is_trained(self) -> bool:
        return self.trained

    def predict_last(self, predictors: pd.DataFrame) -> int:
        """
        Makes a single prediction based on the latest sample in the predictors dataframe.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.

        :return:
            An integer 1, 0, or -1 indicating the prediction:
                1 representing a price increase
                0 representing no significant change in price
                -1 representing a price decrease.
        """
        assert self.is_trained(), "The model has not been trained yet."

        predictors = predictors.iloc[-self.time_step:].to_numpy()

        try:
            predictors = self.scaler.transform(predictors)
            sample = predictors.reshape(1, predictors.shape[0], predictors.shape[1])
            prediction = self.model.predict(sample)
            prediction = np.argmax(prediction, axis=1)[0]

            # Note that an output of 2 indicates that the price is predicted to decrease.
            return -1 if prediction == 2 else prediction

        except ValueError:
            return 0

    def evaluate_accuracy(self, test_predictors: pd.DataFrame, test_labels: pd.Series) -> float:
        """
        Evaluate the model's accuracy with some test data.
        """
        assert self.is_trained(), "The model has not been trained yet."
        predictors, labels = self._preprocess(predictors=test_predictors, labels=test_labels)

        predictions = np.argmax(self.model.predict(predictors), axis=1)
        labels = np.argmax(labels, axis=1)

        return accuracy_score(predictions, labels)

    def predict(self, predictors: pd.DataFrame) -> np.ndarray:
        """
        Makes a prediction for each sample in the input predictors dataframe.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.

        :return:
            A series containing integers 1, 0, or -1 indicating the predictions:
                1 representing a price increase
                0 representing no significant change in price
                -1 representing a price decrease.
        """
        assert self.is_trained(), "The model has not been trained yet."
        predictors = self.scaler.transform(predictors)
        predictors = self._reshape_predictors(predictors)
        predictions = self.model.predict(predictors)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def reset_model(self) -> None:
        self.model = Sequential()
        self.trained = False

    def _preprocess(self, predictors: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses input predictors and labels dataframes/series into a 3D array and a 2D array
        respectively, so as to input them into the LSTM model.

        :param predictors:
            Dataframe of predictor variables, with dates as the index.
        :param labels:
            Series of labels of 0's, 1's or -1's, with dates as the index.

        :return:
            A 3D array of predictor variables with dimensions representing [samples, timesteps, featuers].
            A 2D array of containing the one-hot encoded labels.
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
        reshaped_predictors = self._reshape_predictors(predictors=predictors)
        reshaped_labels = self._reshape_labels(labels=labels)
        return reshaped_predictors, reshaped_labels

    def _reshape_labels(self, labels: np.ndarray) -> np.ndarray:
        # One-hot encode the labels
        reshaped_labels = to_categorical(labels, num_classes=3, dtype=int)

        # Trim the labels to match the predictors array
        # Note that the predictors array will lose the first (self.time_step - 1) rows from the reshaping
        # So, we remove the first (self.time_step - 1) rows from the labels dataframe as well
        reshaped_labels = reshaped_labels[self.time_step - 1:]

        return reshaped_labels

    def _reshape_predictors(self, predictors: np.ndarray) -> np.ndarray:
        assert len(predictors.shape) == 2, "Input array is not 2-dimensional"
        assert len(predictors) >= self.time_step, \
            f"Not enough rows to introduce an additional dimension of size {self.time_step}"

        reshaped_predictors = None

        for index in range(self.time_step, len(predictors) + 1):
            row = predictors[index - self.time_step: index]
            row = row.reshape(1, self.time_step, row.shape[1])
            reshaped_predictors = row if reshaped_predictors is None \
                else np.concatenate((reshaped_predictors, row), axis=0)

        return reshaped_predictors
