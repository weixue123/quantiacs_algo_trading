from typing import List

import pandas as pd

__all__ = ["get_futures_list", "get_settings"]


def get_futures_list(filter_insignificant_lag_1_acf: bool = False) -> List[str]:
    """
    Returns a list of futures' tickers to use for backtesting.

    :param filter_insignificant_lag_1_acf:
        If true, returns only futures with statistically significant lag 1 autocorrelation in their daily returns.
        If false, returns all 88 futures.
    """
    if filter_insignificant_lag_1_acf:
        # Determined with prior research
        return ['F_AD', 'F_C', 'F_DX', 'F_ED', 'F_ES', 'F_FC', 'F_HG', 'F_LB', 'F_LC', 'F_MD', 'F_NG', 'F_NQ', 'F_NR',
                'F_O', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_SB', 'F_TU', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_UB', 'F_LX',
                'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_BG', 'F_LU', 'F_AH', 'F_DZ', 'F_FL', 'F_FM', 'F_FY', 'F_GX', 'F_HP',
                'F_LR', 'F_LQ', 'F_NY', 'F_RF', 'F_SH', 'F_SX', 'F_EB', 'F_VW', 'F_GD', 'F_F']
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
