from models.lgbm.training_util import split, train_model
from systems.systems_util import get_futures_list
from utils.data_loader import load_processed_data

FUTURES_LIST = get_futures_list(filter_insignificant_lag=2)

params = {'num_threads': 4,
          'num_class': 3,
          'objective': 'multiclassova',
          'seed': 123,
          'tree_learner': 'feature',
          'is_unbalance': True}

# tune these
params['learning_rate'] = 0.03
params['boosting_type'] = 'gbdt'
params['metric'] = 'multi_logloss'
params['num_leaves'] = 15
params['extra_trees'] = True
params['sub_feature'] = 0.7
params['min_data'] = 20
params['max_depth'] = 20
params['num_iterations'] = 500
params['max_bin'] = 500

params['num_boost_round'] = 175
params['early_stopping_rounds'] = 30


def train(ticker):
    data = load_processed_data(ticker)
    data = data.loc[:"2020-12-31"]
    data = data.iloc[:-1]
    X_train, X_test, y_train, y_test = split(data)
    model = train_model(X_train, X_test, y_train, y_test, params)
    return model


train(FUTURES_LIST[0])
