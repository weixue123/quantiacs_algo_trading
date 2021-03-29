from utils.data_loader import load_processed_data

from systems.settings import get_futures_list

for index, ticker in enumerate(get_futures_list()):
    print(f"{index}: {ticker}")
    load_processed_data(ticker)
