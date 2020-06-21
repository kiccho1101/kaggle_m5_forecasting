# %%
import pandas as pd
from thunderbolt import Thunderbolt
import sys
import os
from typing import Dict

sys.path.append(os.getcwd() + "/../..")

from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.fe_unemployment import read_unemployment_data

tb = Thunderbolt("./../../resource")
data: pd.DataFrame = tb.get_data("MakeData")

raw: RawData = tb.get_data("LoadRawData")

# %%
import seaborn as sns

stock = pd.read_csv("./../../external_data/stock.csv")
stock.columns = ["date", "close_last", "volume", "open", "high", "low"]
stock["date"] = pd.to_datetime(stock["date"]).dt.strftime("%Y-%m-%d")
for col in ["close_last", "open", "high", "low"]:
    stock[col] = stock[col].map(lambda x: float(x.replace("$", "")))
stock = stock[["date", "close_last", "volume"]]
stock.columns = ["date", "fe_stock_price", "fe_stock_volume"]
stock = raw.calendar[["date"]].merge(stock, on="date", how="left")
stock["fe_stock_price"] = (
    stock["fe_stock_price"].fillna(method="ffill").fillna(method="bfill")
)
stock["fe_stock_volume"] = (
    stock["fe_stock_volume"].fillna(method="ffill").fillna(method="bfill")
)

# %%
import numpy as np

stock["fe_stock_volume"] = np.log1p(stock["fe_stock_volume"])

# raw.calendar["d"] = raw.calendar["d"].apply(lambda d: int(d.replace("d_", "")))

cat_id = 0

df: pd.DataFrame = data[data["cat_id"] == cat_id].groupby(["d", "state_id"])[
    "sales"
].mean().reset_index().merge(raw.calendar[["d", "date"]], on="d", how="left").merge(
    stock, on="date", how="left"
)
df.index = pd.to_datetime(df["date"])

# %%

state_id = 2
sns_plot = sns.pairplot(
    df[df["state_id"] == state_id][["sales", "fe_stock_price", "fe_stock_volume"]]
)
sns_plot.savefig(
    os.path.join(os.path.dirname(__file__), "stock", f"pairplot_{state_id}.png")
)

# %%
stock["fe_stock_volume"]
