# %%
from thunderbolt import Thunderbolt
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + "/../..")

from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.utils import decode_ids


tb = Thunderbolt("./../../resource")
data: pd.DataFrame = tb.get_data("MakeData")
data = decode_ids(data)
raw: RawData = tb.get_data("LoadRawData")

# %%
df = data.groupby("id")[["sell_price", "sales"]].mean().reset_index()


# %%

sell_price = data.groupby("item_id")["sell_price"].mean()

d_cols = [f"d_{d}" for d in range(1, 1942)]
date_idx = pd.to_datetime(raw.calendar["date"][:1941])
for i, s in enumerate(
    raw.sales_train_validation[raw.sales_train_validation.cat_id == "FOODS"][
        ["item_id"] + d_cols
    ]
    .groupby("item_id")
    .sum()
    .reset_index()
    .values
):
    if i >= 150 and i <= 180:
        item_id = s[0]
        sales = s[1:]
        print(i, item_id, "price:", sell_price[item_id])
        pd.Series(sales, index=date_idx).plot()
        plt.show()


# %%
for store_id in data["store_id"].unique():
    print(store_id)
    d = (
        data[(data["cat_id"] == "FOODS") & (data["store_id"] == store_id)]
        .groupby("d")["sales"]
        .mean()
    )
    d.index = pd.to_datetime(raw.calendar["date"])
    d.plot()
    plt.show()


# %%
[
    int(d.replace("d_", ""))
    for d in raw.calendar[raw.calendar["date"].str[5:] == "01-01"]["d"].values
]


# %%
chinise_new_year_days = [
    ["2011-02-03", "2011-02-10"],
    ["2012-01-23", "2012-01-30"],
    ["2013-02-10", "2013-02-17"],
    ["2014-01-31", "2014-02-07"],
    ["2015-02-19", "2015-02-26"],
    ["2016-02-08", "2016-02-15"],
]
nba_finals_dates = [
    "2011-05-31",
    "2011-06-02",
    "2011-06-05",
    "2011-06-07",
    "2011-06-09",
    "2011-06-12",
    "2012-06-12",
    "2012-06-14",
    "2012-06-17",
    "2012-06-19",
    "2012-06-21",
    "2013-06-06",
    "2013-06-09",
    "2013-06-11",
    "2013-06-13",
    "2013-06-16",
    "2013-06-18",
    "2013-06-20",
    "2014-06-05",
    "2014-06-08",
    "2014-06-10",
    "2014-06-12",
    "2014-06-15",
    "2015-06-04",
    "2015-06-07",
    "2015-06-09",
    "2015-06-11",
    "2015-06-14",
    "2015-06-16",
    "2016-06-02",
    "2016-06-05",
    "2016-06-08",
    "2016-06-10",
    "2016-06-13",
    "2016-06-16",
    "2016-06-19",
]
pentecost_dates = [
    "2011-06-12",
    "2012-05-27",
    "2013-05-19",
    "2014-06-08",
    "2015-05-24",
    "2016-05-15",
]
orthodox_pentecost_dates = [
    "2011-06-12",
    "2012-06-03",
    "2013-06-23",
    "2014-06-08",
    "2015-05-31",
    "2016-06-19",
]


# %%
