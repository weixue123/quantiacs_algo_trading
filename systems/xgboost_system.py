import pickle
import numpy as np
import os
import sys
from pathlib import Path

project_root_path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(project_root_path) + "\..")

from data_processing.data_processor import DataProcessor
from systems.systems_util import build_ohclv_dataframe


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    print(f"Testing: {DATE[-1]}")  # debug
    nMarkets = len(settings["markets"])
    lookback = settings["lookback"]
    positions = np.zeros(nMarkets, dtype=np.float)

    for index, ticker in enumerate(settings["markets"]):
        try:
            ohclv_data = build_ohclv_dataframe(
                DATE, OPEN, HIGH, LOW, CLOSE, VOL, index
            )  # build dataframe
            predictors = DataProcessor(
                data=ohclv_data
            ).build_predictors()  # create inputs
            predictors = predictors.iloc[-1:]  # get data from the last time frame
            # model = pickle.load(
            #     open(f"./models/xgboost/trained_xgboost_models/{ticker}.pkl", "rb")
            # )
            model = load_model(ticker)
            prediction = model.predict(
                predictors.to_numpy()
            )  # to_numpy for xgboost model
            positions[index] = prediction[0]

        # for NaN data or futures that will not be traded, set position to 0
        except:
            positions[index] = 0

    print(positions)
    return positions, settings


def mySettings():

    settings = {}
    settings["markets"] = [
        "CASH",
        "F_AD",
        "F_BO",
        "F_BP",
        "F_C",
        "F_CC",
        "F_CD",
        "F_CL",
        "F_CT",
        "F_DX",
        "F_EC",
        "F_ED",
        "F_ES",
        "F_FC",
        "F_FV",
        "F_GC",
        "F_HG",
        "F_HO",
        "F_JY",
        "F_KC",
        "F_LB",
        "F_LC",
        "F_LN",
        "F_MD",
        "F_MP",
        "F_NG",
        "F_NQ",
        "F_NR",
        "F_O",
        "F_OJ",
        "F_PA",
        "F_PL",
        "F_RB",
        "F_RU",
        "F_S",
        "F_SB",
        "F_SF",
        "F_SI",
        "F_SM",
        "F_TU",
        "F_TY",
        "F_US",
        "F_W",
        "F_XX",
        "F_YM",
        "F_AX",
        "F_CA",
        "F_DT",
        "F_UB",
        "F_UZ",
        "F_GS",
        "F_LX",
        "F_SS",
        "F_DL",
        "F_ZQ",
        "F_VX",
        "F_AE",
        "F_BG",
        "F_BC",
        "F_LU",
        "F_DM",
        "F_AH",
        "F_CF",
        "F_DZ",
        "F_FB",
        "F_FL",
        "F_FM",
        "F_FP",
        "F_FY",
        "F_GX",
        "F_HP",
        "F_LR",
        "F_LQ",
        "F_ND",
        "F_NY",
        "F_PQ",
        "F_RR",
        "F_RF",
        "F_RP",
        "F_RY",
        "F_SH",
        "F_SX",
        "F_TR",
        "F_EB",
        "F_VF",
        "F_VT",
        "F_VW",
        "F_GD",
        "F_F",
    ]

    settings["beginInSample"] = "20190101"  # 20210101
    settings["endInSample"] = "20210331"  # 20210331
    settings["lookback"] = 504
    settings["budget"] = 10 ** 6
    settings["slippage"] = 0.05

    return settings


def load_model(ticker: str):
    """
    Helper function to load a trained model previously saved as a pickle.
    """
    storage_dir = (
        Path(os.path.dirname(__file__)).parent / "models/xgboost/trained_xgboost_models"
    )
    pickle_in = open(f"{storage_dir}\\{ticker}.pkl", "rb")
    return pickle.load(pickle_in)


if __name__ == "__main__":
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)