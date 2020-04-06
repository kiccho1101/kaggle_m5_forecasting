# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
from kaggle_m5_forecasting import RawData
from kaggle_m5_forecasting.utils import (
    timer,
    merge_by_concat,
    print_mem_usage,
    reduce_mem_usage,
)
from thunderbolt import Thunderbolt
from tqdm import tqdm

tb = Thunderbolt("./resource")
with timer("load"):
    raw: RawData = tb.get_data("LoadRawData")
    # data: pd.DataFrame = tb.get_data("FEBasic")
    # model: lgb.train = tb.get_data("LGBMVal")


# %%
df: pd.DataFrame = raw.sales_train_validation.set_index(["store_id", "item_id"])

# %%
