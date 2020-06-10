# %%
import pandas as pd
from thunderbolt import Thunderbolt
from kaggle_m5_forecasting import RawData

tb = Thunderbolt("./resource")

raw: RawData = tb.get_data("LoadRawData")


# %%
raw.sales_train_validation
