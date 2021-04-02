from typing import List, Union

import numpy as np
import pandas as pd

__all__ = ["get_futures_list", "get_settings", "normalize_weights", "build_ohclv_dataframe"]


def get_futures_list(filter_insignificant_lag_1_acf: bool = False) -> List[str]:
    """
    Returns a list of futures' tickers to use for backtesting.

    :param filter_insignificant_lag_1_acf:
        If true, returns only futures with statistically significant lag 1 autocorrelation in their daily returns.
        If false, returns all 88 futures.
    """
    if filter_insignificant_lag_1_acf:
        # Determined with prior research
        # return ['F_ED', 'F_ES', 'F_NQ', 'F_O', 'F_SB', 'F_LX', 'F_SS', 'F_VX', 'F_LU', 'F_LQ', 'F_RF', 'F_EB', 'F_F']
        return ['F_AD', 'F_C', 'F_DX', 'F_ED', 'F_ES', 'F_FC', 'F_HG', 'F_LB', 'F_LC', 'F_MD', 'F_NG', 'F_NQ',
                'F_NR', 'F_O', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_SB', 'F_TU', 'F_XX', 'F_YM', 'F_AX', 'F_CA',
                'F_UB', 'F_LX', 'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_BG', 'F_LU', 'F_AH', 'F_DZ', 'F_FL', 'F_FM',
                'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_NY', 'F_RF', 'F_SH', 'F_SX', 'F_EB', 'F_VW', 'F_GD',
                'F_F']
    else:
        return ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC',
                'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ',
                'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU',
                'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ', 'F_GS', 'F_LX', 'F_SS',
                'F_DL', 'F_ZQ', 'F_VX', 'F_AE', 'F_BG', 'F_BC', 'F_LU', 'F_DM', 'F_AH', 'F_CF', 'F_DZ', 'F_FB', 'F_FL',
                'F_FM', 'F_FP', 'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_ND', 'F_NY', 'F_PQ', 'F_RR', 'F_RF', 'F_RP',
                'F_RY', 'F_SH', 'F_SX', 'F_TR', 'F_EB', 'F_VF', 'F_VT', 'F_VW', 'F_GD', 'F_F']


def get_settings(first_date: str = "20190123", last_date: str = "20210331"):
    """
    Obtains the base settings to use for any backtesting.

    :param first_date:
        Date string in YYYYMMDD format indicating the first date of the backtest
    :param last_date:
        Date string in YYYYMMDD format indicating the last date of the backtest

    :return:
        Full set of settings with the full futures list.
    """

    try:
        assert pd.to_datetime(first_date, format="%Y%m%d") < pd.to_datetime(last_date, format="%Y%m%d"), \
            "Last date must be later than first date."
    except ValueError:
        raise ValueError("Input date strings should be in the YYYYMMDD format")

    # Note that setting the beginInSample date to 20190123 is equivalent to setting the backtesting's start
    # date to 2021-01-04 (first day of 2021).

    return {'lookback': 504,
            'budget': 10 ** 6,
            'slippage': 0.05,
            'beginInSample': first_date,
            'endInSample': last_date}


def normalize_weights(weights: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Helper function to normalize an input list or array of weights so that their sum is 1.
    """
    weights = np.array(weights)
    assert len(weights.shape) == 1, "Weights should be a 1-D array."

    total_weights = np.nansum(np.abs(weights))

    # If total weights is zero, adjust the weights for cash (index 0) to 1; this avoids division by zero
    if total_weights == 0:
        weights[0] = 1
        total_weights = 1

    return weights / total_weights


def build_ohclv_dataframe(DATE: List[int], OPEN: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, CLOSE: np.ndarray,
                          VOL: np.ndarray, index: int):
    """
    Helper function to build a dataframe of open, high, low, close and volume data from the arrays given by
    myTradingSystem
    """
    dates: List[pd.datetime] = list(map(lambda date_int: pd.to_datetime(date_int, format="%Y%m%d"), DATE))
    ohclv_data = pd.DataFrame(index=dates)
    ohclv_data["OPEN"] = OPEN[:, index]
    ohclv_data["HIGH"] = HIGH[:, index]
    ohclv_data["LOW"] = LOW[:, index]
    ohclv_data["CLOSE"] = CLOSE[:, index]
    ohclv_data["VOL"] = VOL[:, index]
    return ohclv_data
