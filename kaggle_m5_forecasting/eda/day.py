# %%
from thunderbolt import Thunderbolt
import pandas as pd
import sys
import os
import sklearn.preprocessing
from typing import Dict
import pickle

sys.path.append(os.getcwd() + "/../..")

from kaggle_m5_forecasting.data.load_data import RawData
from tqdm import tqdm


def decode_ids(data: pd.DataFrame) -> pd.DataFrame:

    cat_encoders: Dict[str, sklearn.preprocessing.LabelEncoder] = pickle.load(
        open("./../../cat_encoders.pkl", "rb")
    )
    for col in tqdm(["item_id", "dept_id", "cat_id", "store_id", "state_id"]):
        data[col] = data[col].apply(lambda x: cat_encoders[col].classes_[x])
    return data


tb = Thunderbolt("./../../resource")
data: pd.DataFrame = tb.get_data("MakeData")
data = decode_ids(data)
raw: RawData = tb.get_data("LoadRawData")

# %%
df = data.groupby("id")[["sell_price", "sales"]].mean().reset_index()


# %%
import matplotlib.pyplot as plt

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
d = data.groupby("id")["sales_is_zero"].mean()


# %%
sel_idx = d[d > 0.97].index[3]
raw.sales_train_validation.set_index("id").loc[sel_idx, :].filter(like="d_").plot()

# %%
d[d > 0.97]
