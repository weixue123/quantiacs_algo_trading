from typing import Tuple

import numpy as np

from systems.systems_util import get_futures_list, get_settings


def myTradingSystem(settings) -> Tuple[np.ndarray, dict]:
    """
    Base trading system that simply longs every asset with equal weight.
    """
    weights = []
    for asset in settings["markets"]:
        weights.append(0) if asset == "CASH" else weights.append(1)
    weights = np.array(weights)

    return weights, settings


def mySettings():
    settings = get_settings()
    settings["markets"] = ["CASH", *get_futures_list()]
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
