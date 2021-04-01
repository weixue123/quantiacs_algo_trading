import os
from pathlib import Path
from typing import Tuple

import json


def load_arima_parameters(ticker: str) -> Tuple[int, int, int]:
    print(f"Loading ARIMA parameters for {ticker}")
    storage_dir = Path(os.path.dirname(__file__)) / "params"
    with open(f"{storage_dir}/{ticker}_params.txt") as f:
        params = json.load(f)
    return params["p"], params["d"], params["q"]
