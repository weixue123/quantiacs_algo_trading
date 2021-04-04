import lightgbm as lgb
import numpy as np

from sklearn.model_selection import train_test_split


def split(data):
    labels = data['LABELS']
    X = data.copy().drop([
        "LABELS",
    ], axis=1)
    print(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.33, random_state=123, shuffle=False)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, params):
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    labels = train_data.get_label()
    print(np.unique(labels))

    num_boost_round = params['num_boost_round']
    early_stopping_rounds = params['early_stopping_rounds']

    model = lgb.train(params=params, train_set=train_data,
                      valid_sets=[test_data],
                      num_boost_round=num_boost_round,
                      early_stopping_rounds=early_stopping_rounds)

    return model
