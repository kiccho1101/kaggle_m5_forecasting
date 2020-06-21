# %%
import pandas as pd
from thunderbolt import Thunderbolt
from kaggle_m5_forecasting import RawData

tb = Thunderbolt("./resource")
data: pd.DataFrame = tb.get_data("MakeData")
raw: RawData = tb.get_data("LoadRawData")
