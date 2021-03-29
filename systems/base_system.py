from typing import Tuple

import numpy as np

from systems.settings import get_settings


def myTradingSystem(settings) -> Tuple[np.ndarray, dict]:
    """
    Base system that longs every asset with equal weight.
    """
    weights = []
    for asset in settings["markets"]:
        weights.append(0) if asset == "CASH" else weights.append(1)
    weights = np.array(weights)

    return weights, settings


def mySettings():
    settings = get_settings()
    return settings


if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
