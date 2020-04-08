from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


class FERollingMean(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling mean"):
            for lag in [28]:
                for w_size in tqdm([7, 30, 60, 90, 180]):
                    data[f"rolling_mean_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                        .astype(np.float16)
                    )
                    data[f"rolling_mean_item_t{lag}_{w_size}"] = (
                        data.groupby(["item_id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                        .astype(np.float16)
                    )
        df = data.filter(like="rolling_mean")
        print(df.info())
        self.dump(df)


class FERollingStd(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling features"):
            for lag in [28]:
                for w_size in tqdm([7, 30, 60, 90, 180]):
                    data[f"rolling_std_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).std())
                        .astype(np.float16)
                    )
                    data[f"rolling_std_item_t{lag}_{w_size}"] = (
                        data.groupby(["item_id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).std())
                        .astype(np.float16)
                    )
        df = data.filter(like="rolling_std_")
        print(df.info())
        self.dump(df)


class FERollingSkew(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling features"):
            for lag in [28]:
                for w_size in tqdm([30]):
                    data[f"rolling_skew_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).skew())
                        .astype(np.float16)
                    )
        df = data.filter(like="rolling_skew_t")
        print(df.info())
        self.dump(df)


class FERollingKurt(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling features"):
            for lag in [28]:
                for w_size in tqdm([30]):
                    data[f"rolling_kurt_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).kurt())
                        .astype(np.float16)
                    )
        df = data.filter(like="rolling_kurt_t")
        print(df.info())
        self.dump(df)
