# XGBoost
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import hp
from hyperopt.pyll import scope

# utils & systems
import os
import sys
from pathlib import Path

project_root_path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(project_root_path) + "\..\..")

from utils.data_loader import load_processed_data
from systems.systems_util import get_futures_list
from models.xgboost.training_util import hyperopt, train_model


# Cross validation
ts_crossval = TimeSeriesSplit(n_splits=5)

# Define search space for bayesian optimisation
XGB_param_hyperopt = {
    "booster": hp.choice("booster", ["gblinear"]),
    "max_depth": scope.int(hp.quniform("max_depth", 1, 5, 1)),
    "subsample": hp.uniform("subsample", 0.4, 0.6),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.6),
    "colsample_bynode": hp.uniform("colsample_bynode", 0.4, 0.6),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0.4, 0.6),
    "gamma": hp.uniform("gamma", 0, 10),
    "min_child_weight": hp.uniform("min_child_weight", 1.5, 2.3),
    "n_estimators": 100,
    "reg_lambda": hp.uniform("reg_lambda", 1, 8),
    "reg_alpha": hp.uniform("reg_alpha", 0, 0.02),
}

futures = get_futures_list(filter_insignificant_lag_1_acf=True)

# Pre-train models and save the weights
for ticker in futures:
    data = load_processed_data(ticker)
    data = data.loc[:"2020-12-31"]
    data = data.iloc[:-1]
    train_model(data,XGB_param_hyperopt,ts_crossval,ticker)
