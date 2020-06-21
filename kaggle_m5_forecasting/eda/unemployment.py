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

unemployment = read_unemployment_data(
    date_range=raw.calendar[["date"]], external_data_path="./../../external_data"
)

raw.calendar["d"] = raw.calendar["d"].apply(lambda d: int(d.replace("d_", "")))

cat_id = 0

df: pd.DataFrame = data[data["cat_id"] == cat_id].groupby(["d", "state_id"])[
    "sales"
].mean().reset_index().merge(raw.calendar[["d", "date"]], on="d", how="left").merge(
    unemployment, on=["date", "state_id"], how="left"
)
df.index = pd.to_datetime(df["date"])

# %%

state_id = 1
sns_plot = sns.pairplot(df[df["state_id"] == state_id][["sales", "fe_unemployment"]])
sns_plot.savefig(
    os.path.join(os.path.dirname(__file__), "unemployment", f"pairplot_{state_id}.png")
)

# %%
unemployment
