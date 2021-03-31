import os
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from models.lstm.lstm_model import LSTMModel
from utils.data_loader import load_processed_data

__all__ = ["get_training_data", "get_cross_validation_results", "build_optimized_model", "save_model"]


def get_training_data(ticker: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the processed data corresponding to the input ticker.
    Removes any rows that are supposed to be used for testing (i.e. data from Jan 2021 onwards).
    Splits the training data into predictors and labels
    """
    data = load_processed_data(ticker)
    data = data.loc[:"2020-12-31"]

    # Drop last row because it would have had 2021's first day's change in price
    data = data.iloc[:-1]

    predictors, labels = data.drop("LABELS", axis=1), data["LABELS"]
    return predictors, labels


def get_cross_validation_results(predictors: pd.DataFrame, labels: pd.Series) -> pd.DataFrame():
    """
    Carries out cross-validaiton. Returns a dataframe that records the cross-validation
    accuracies along with the hyperparameters combintaions that produced them.

    The output dataframe will have the following four columns:
        1. CV Accuracy
        2. Epochs
        3. Time-Step Per Sample
        4. Hidden Layers
    """
    iteration_number = 1
    cv_accuracy_records = []

    # Hard-coded combinations of hyperparameters to iterate through
    # Hard-coded cross-validation splits
    for time_step in [1, 3]:
        for epochs in [50, 100]:
            for hidden_layers in [1, 2]:
                print(f"Cross-validating combination {iteration_number}/8...")

                accuracy_sum = 0

                # Train with 0%-25%, test on 25%-50%
                X_train, X_remaining, y_train, y_remaining = train_test_split(predictors, labels, test_size=0.75,
                                                                              shuffle=False)
                X_test, _, y_test, _ = train_test_split(X_remaining, y_remaining, test_size=0.66, shuffle=False)
                model = LSTMModel(time_step=time_step)
                model.build_and_train_model(X_train, y_train, epochs=epochs, hidden_layers=hidden_layers)
                accuracy_sum += model.evaluate_accuracy(X_test, y_test)

                # Train with 0%-50%, test on 50%-75%
                X_train, X_remaining, y_train, y_remaining = train_test_split(predictors, labels, test_size=0.5,
                                                                              shuffle=False)
                X_test, _, y_test, _ = train_test_split(X_remaining, y_remaining, test_size=0.5, shuffle=False)
                model = LSTMModel(time_step=time_step)
                model.build_and_train_model(X_train, y_train, epochs=epochs, hidden_layers=hidden_layers)
                accuracy_sum += model.evaluate_accuracy(X_test, y_test)

                # Train with 0%-75%, test on 75%-100%
                X_train, X_test, y_train, y_test = train_test_split(predictors, labels, test_size=0.25, shuffle=False)
                model = LSTMModel(time_step=time_step)
                model.build_and_train_model(X_train, y_train, epochs=epochs, hidden_layers=hidden_layers)
                accuracy_sum += model.evaluate_accuracy(X_test, y_test)

                cv_accuracy = accuracy_sum / 3

                cv_accuracy_records.append({"CV Accuracy": round(cv_accuracy, 3),
                                            "Epochs": epochs,
                                            "Time-Step Per Sample": time_step,
                                            "Hidden Layers": hidden_layers})

                iteration_number += 1

    return pd.DataFrame(cv_accuracy_records)


def build_optimized_model(predictors: pd.DataFrame, labels: pd.Series, cv_accuracy_records: pd.DataFrame) -> LSTMModel:
    """
    Given a dataframe of cross-validation results, build the optimized model.
    """
    top_hyperparameters = cv_accuracy_records.sort_values("CV Accuracy", ascending=False).iloc[0]
    time_step = int(top_hyperparameters["Time-Step Per Sample"])
    epochs = int(top_hyperparameters["Epochs"])
    hidden_layers = int(top_hyperparameters["Hidden Layers"])

    model = LSTMModel(time_step=time_step)
    model.build_and_train_model(predictors=predictors, labels=labels, epochs=epochs, hidden_layers=hidden_layers)
    return model


def save_model(ticker: str, model: LSTMModel):
    """
    Saves an LSTM model to a pickle file.
    """
    storage_dir = Path(os.path.dirname(__file__)) / "serialized_models"
    pickle_out = open(f"{storage_dir}/{ticker}.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()
