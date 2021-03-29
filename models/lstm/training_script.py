from models.lstm.training_util import get_training_data, get_cross_validation_results, build_optimized_model, save_model
from systems.systems_util import get_futures_list

futures_list = get_futures_list(filter_insignificant_lag_1_acf=True)

for ticker in futures_list:
    print(f"Training model for {ticker}")

    predictors, labels = get_training_data(ticker)
    cv_accuracy_records = get_cross_validation_results(predictors=predictors, labels=labels)
    model = build_optimized_model(predictors=predictors, labels=labels, cv_accuracy_records=cv_accuracy_records)
    save_model(ticker, model)

    print(f"Trained and saved model for {ticker}!\n")
