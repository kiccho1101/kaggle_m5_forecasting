# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from thunderbolt import Thunderbolt
from kaggle_m5_forecasting.utils import timer
import sys
import os

sys.path.append(os.getcwd() + "/../..")

tb = Thunderbolt("./../../resource")
data: pd.DataFrame = tb.get_data("MakeData")

# %%

with timer("calc rolling_store_id_cat_id_mean"):
    lag = 28
    w_size = 30
    data["fe_rolling_store_id_cat_id_mean"] = data.groupby(["store_id", "cat_id"])[
        "sales"
    ].transform(lambda x: x.shift(lag).rolling(w_size).mean())
