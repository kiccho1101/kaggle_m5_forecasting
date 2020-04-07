from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


class FEPriceChange(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        with timer("make price lag_1 features"):
            data["lag_price_t1"] = data.groupby(["id"])["sell_price"].transform(
                lambda x: x.shift(1)
            )
            data["fe_price_change_t1"] = (data["lag_price_t1"] - data["sell_price"]) / (
                data["lag_price_t1"]
            )
            data.drop("lag_price_t1", axis=1, inplace=True)

        with timer("make price lag_365 features"):
            data["rolling_price_max_t365"] = data.groupby(["id"])[
                "sell_price"
            ].transform(lambda x: x.shift(1).rolling(365).max())
            data["fe_price_change_t365"] = (
                data["rolling_price_max_t365"] - data["sell_price"]
            ) / (data["rolling_price_max_t365"])
            data.drop("rolling_price_max_t365", axis=1, inplace=True)

        df = data.filter(like="fe_price_change")
        print(df.info())
        self.dump(df)
