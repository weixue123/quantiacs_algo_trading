from pathlib import Path
import os
import pickle
import warnings

import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

__all__ = ["train_model", "load_xgb_model"]


def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval, classifier, cross_val):
    """
    Perform bayesian optimisation based on the provided data and hyperparameter space

    Input:
        1. Hyperparameter space
        2. Training feature
        3. Training label
        4. Validation feature
        5. Validation label
        6. Number of iteration
        7. Type of classifier
        8. Cross validation object
    Output:
        1. Training history
        2. Best set of hyperparameters
    """

    def objective_function(params):
        clf = classifier(**params, random_state=4013)
        score = cross_val_score(clf, X_train, y_train, cv=cross_val).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, param_space, algo=tpe.suggest, max_evals=100,
                      trials=trials, rstate=np.random.RandomState(1))
    best_param_values = best_param
    return trials, best_param_values


def train_model(data, param_space, cross_val, ticker):
    # Split test-train set
    train_data, eval_data = train_test_split(data, test_size=0.2, shuffle=False)  # do not shuffle time series data
    train_predictors, train_labels = train_data.drop("LABELS", axis=1), train_data["LABELS"]
    eval_predictors, eval_labels = eval_data.drop("LABELS", axis=1), eval_data["LABELS"]
    X_train = train_predictors.to_numpy()
    y_train = train_labels.to_numpy()
    X_eval = eval_predictors.to_numpy()
    y_eval = eval_labels.to_numpy()

    # Bayesian Optimisation
    XGB_hyperopt = hyperopt(param_space, X_train, y_train, X_eval, y_eval, 100, XGBClassifier, cross_val)

    # Obtain optimal hyperparameters
    xg_chosen = XGB_hyperopt[1]  # params of the best model

    # Retrain with optimal hyperparameters
    xgboost_model = XGBClassifier(booster=['gbtree', 'gblinear', 'dart'][xg_chosen['booster']],
                                  max_depth=int(xg_chosen['max_depth']), subsample=xg_chosen['subsample'],
                                  colsample_bytree=xg_chosen['colsample_bytree'],
                                  colsample_bynode=xg_chosen['colsample_bynode'],
                                  colsample_bylevel=xg_chosen['colsample_bylevel'],
                                  n_estimators=100, reg_alpha=xg_chosen['reg_alpha'],
                                  reg_lambda=xg_chosen['reg_lambda'], gamma=xg_chosen['gamma'],
                                  min_child_weight=xg_chosen['min_child_weight'], random_state=4013)

    # Train the model
    xgboost_model.fit(X_train, y_train, verbose=False, eval_metric='mlogloss')
    pickle.dump(xgboost_model, open(f"./models/xgboost/trained_xgboost_models/{ticker}.pkl", "wb"))
    print(f"Successfully trained XGBoost model for {ticker}")  # debug


def load_xgb_model(ticker: str) -> XGBClassifier:
    """
    Helper function to load a trained model previously saved as a pickle.
    """
    print(f"Loading XGBoost Model for {ticker}")
    storage_dir = (Path(os.path.dirname(__file__)) / "trained_xgboost_models")
    pickle_in = open(f"{storage_dir}/{ticker}.pkl", "rb")
    return pickle.load(pickle_in)
