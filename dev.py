# %%
from kaggle_m5_forecasting import Data
from thunderbolt import Thunderbolt

tb = Thunderbolt("./resource")
d: Data = tb.get_data("LoadData")


# %%
d.sell_prices.shape
