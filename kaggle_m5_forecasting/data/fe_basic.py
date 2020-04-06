from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


class FEBasic(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        with timer("make shift features"):
            for days in tqdm(list(range(28, 35)) + list(range(338, 340))):
                data[f"shift_t{days}"] = (
                    data.groupby(["id"])["sales"]
                    .transform(lambda x: x.shift(days))
                    .astype(np.float16)
                )
            print_mem_usage(data)

        with timer("make rolling features"):
            for days in tqdm([7, 30, 60, 90, 180]):
                data[f"rolling_mean_t{days}"] = (
                    data.groupby(["id"])["sales"]
                    .transform(lambda x: x.shift(28).rolling(days).mean())
                    .astype(np.float16)
                )
                data[f"rolling_std_t{days}"] = (
                    data.groupby(["id"])["sales"]
                    .transform(lambda x: x.shift(28).rolling(days).std())
                    .astype(np.float16)
                )
                data[f"rolling_max_t{days}"] = (
                    data.groupby(["id"])["sales"]
                    .transform(lambda x: x.shift(28).rolling(days).max())
                    .astype(np.float16)
                )
                data[f"rolling_min_t{days}"] = (
                    data.groupby(["id"])["sales"]
                    .transform(lambda x: x.shift(28).rolling(days).min())
                    .astype(np.float16)
                )
            data[f"rolling_skew_t30"] = (
                data.groupby(["id"])["sales"]
                .transform(lambda x: x.shift(28).rolling(30).skew())
                .astype(np.float16)
            )
            data[f"rolling_kurt_t30"] = (
                data.groupby(["id"])["sales"]
                .transform(lambda x: x.shift(28).rolling(30).kurt())
                .astype(np.float16)
            )
            print_mem_usage(data)

        with timer("make price lag_1 features"):
            data["lag_price_t1"] = data.groupby(["id"])["sell_price"].transform(
                lambda x: x.shift(1)
            )
            data["price_change_t1"] = (data["lag_price_t1"] - data["sell_price"]) / (
                data["lag_price_t1"]
            )
            data.drop("lag_price_t1", axis=1, inplace=True)
            print_mem_usage(data)

        with timer("make price lag_365 features"):
            data["rolling_price_max_t365"] = data.groupby(["id"])[
                "sell_price"
            ].transform(lambda x: x.shift(1).rolling(365).max())
            data["price_change_t365"] = (
                data["rolling_price_max_t365"] - data["sell_price"]
            ) / (data["rolling_price_max_t365"])
            data.drop("rolling_price_max_t365", axis=1, inplace=True)
            print_mem_usage(data)

        with timer("make rolling_price_std_t7"):
            data["rolling_price_std_t7"] = data.groupby(["id"])["sell_price"].transform(
                lambda x: x.rolling(7).std()
            )
            print_mem_usage(data)

        with timer("make rolling_price_std_t30"):
            data["rolling_price_std_t30"] = data.groupby(["id"])[
                "sell_price"
            ].transform(lambda x: x.rolling(30).std())
            print_mem_usage(data)

        self.dump(data)
