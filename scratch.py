import os

import pandas as pd

from utils.indicators import atr

os.chdir("/Users/weixue/Desktop/BT4013/Project/")

data = pd.read_csv(f"systems/tickerData/F_ES.txt", delimiter=",")
data.columns = list(map(lambda col: col.strip(), data.columns))
data = data.set_index("DATE")
data.index = pd.to_datetime(data.index, format="%Y%m%d")

atr(close=data["CLOSE"], high=data["HIGH"], low=data["LOW"])

print(data)
