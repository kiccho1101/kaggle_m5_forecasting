# %%
import datetime
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from thunderbolt import Thunderbolt
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.load_data import RawData

tb = Thunderbolt("./resource")
raw: RawData = tb.get_data("LoadRawData")
# data: pd.DataFrame = tb.get_data("CombineFeatures")
