# %%
import pandas as pd
from thunderbolt import Thunderbolt
import sys
import os

sys.path.append(os.getcwd() + "/../..")

from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.data.fe_weather import read_weather_data
from kaggle_m5_forecasting.utils import decode_ids

tb = Thunderbolt("./../../resource")
data: pd.DataFrame = pd.concat(
    [tb.get_data("MakeData"), tb.get_data("FEWeather")], axis=1
)
weather = read_weather_data("./../../external_data")
weather["date"] = pd.to_datetime(weather["date_time"]).dt.strftime("%Y-%m-%d")
weather.index = pd.to_datetime(weather["date_time"])
weather.index.name = None
weather.drop("date_time", axis=1, inplace=True)

raw: RawData = tb.get_data("LoadRawData")


# %%
raw.calendar["d"] = raw.calendar["d"].apply(lambda d: int(d.replace("d_", "")))
cat_id = 0
df: pd.DataFrame = data[data["cat_id"] == cat_id].groupby(["d", "state_id"])[
    "sales"
].mean().reset_index().merge(raw.calendar[["d", "date"]], on="d", how="left").merge(
    weather, on=["date", "state_id"], how="left"
)
df.index = pd.to_datetime(df["date"])


# %%
import seaborn as sns

selected_cols = [
    "sales",
    "fe_weather_maxtempC",
    "fe_weather_mintempC",
    "fe_weather_sunHour",
    "fe_weather_DewPointC",
    "fe_weather_FeelsLikeC",
    "fe_weather_HeatIndexC",
    "fe_weather_WindChillC",
    "fe_weather_WindGustKmph",
    "fe_weather_cloudcover",
    "fe_weather_humidity",
    "fe_weather_precipMM",
    "fe_weather_pressure",
]
state_id = 2

sns_plot = sns.pairplot(df[df["state_id"] == state_id][selected_cols])
sns_plot.savefig(
    os.path.join(os.path.dirname(__file__), "weather", f"pairplot_{state_id}.png")
)
# %%

corr = df[df["state_id"] == state_id][selected_cols].corr()
sns_plot = sns.heatmap(corr)
sns_plot.get_figure().savefig(
    os.path.join(os.path.dirname(__file__), "weather", f"corr_heatmap_{state_id}.png")
)


# %%
