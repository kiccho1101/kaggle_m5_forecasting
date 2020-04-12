from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


class FERollingSum(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling mean"):
            for lag in [0, 15]:
                for w_size in tqdm([30]):
                    data[f"fe_rolling_sum_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).sum())
                        .astype(np.float16)
                    )
        df = data.filter(like="fe_rolling_sum")
        print(df.info())
        self.dump(df)


class FERollingMean(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling mean"):
            for lag in [7, 14, 28]:
                for w_size in tqdm([7, 30, 60, 90, 180]):
                    data[f"fe_rolling_mean_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                        .astype(np.float16)
                    )
        df = data.filter(like="fe_rolling_mean")
        print(df.info())
        self.dump(df)


class FERollingMeanDW(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling mean of dw"):
            for lag in [7, 28]:
                for w_size in tqdm([7, 30, 60, 90, 180]):
                    data[f"fe_rolling_mean_dw_t{lag}_{w_size}"] = (
                        data.groupby(["id", "tm_dw"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                        .astype(np.float16)
                    )
                    data[f"fe_rolling_mean_dw_store_t{lag}_{w_size}"] = (
                        data.groupby(["store_id", "tm_dw"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                        .astype(np.float16)
                    )
        df = data.filter(like="fe_rolling_mean_dw")
        print(df.info())
        self.dump(df)


class FERollingStd(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling features"):
            for lag in [7, 14, 28]:
                for w_size in tqdm([7, 30, 60, 90, 180]):
                    data[f"fe_rolling_std_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).std())
                        .astype(np.float16)
                    )
        df = data.filter(like="fe_rolling_std_")
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
                    data[f"fe_rolling_skew_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).skew())
                        .astype(np.float16)
                    )
        df = data.filter(like="fe_rolling_skew")
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
                    data[f"fe_rolling_kurt_t{lag}_{w_size}"] = (
                        data.groupby(["id"])["sales"]
                        .transform(lambda x: x.shift(lag).rolling(w_size).kurt())
                        .astype(np.float16)
                    )
        df = data.filter(like="fe_rolling_kurt")
        print(df.info())
        self.dump(df)
