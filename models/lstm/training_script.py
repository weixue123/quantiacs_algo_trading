from models.lstm.training_util import get_training_data, get_cross_validation_results, build_optimized_model, \
    save_model, is_model_trained_and_saved
from systems.systems_util import get_futures_list


def train_and_save_model(ticker: str) -> None:
    print(f"Training model for {ticker}")

    predictors, labels = get_training_data(ticker)
    cv_accuracy_records = get_cross_validation_results(predictors=predictors, labels=labels)
    model = build_optimized_model(predictors=predictors, labels=labels, cv_accuracy_records=cv_accuracy_records)
    save_model(ticker, model)

    print(f"Trained and saved model for {ticker}!\n")


def execute():
    for ticker in get_futures_list(filter_insignificant_lag=1):
        if not is_model_trained_and_saved(ticker=ticker):
            train_and_save_model(ticker=ticker)
        else:
            print(f"Model for {ticker} is already trained and saved!")


if __name__ == '__main__':
    execute()
