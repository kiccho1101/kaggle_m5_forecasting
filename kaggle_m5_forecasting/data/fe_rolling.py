from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import (
    timer,
    reduce_mem_usage,
    print_mem_usage,
    df_parallelize_run,
)
from typing import List, Any
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


def make_lag_roll(LAG_WSIZE: List[Any]):
    data: pd.DataFrame = LAG_WSIZE[0]
    lag = LAG_WSIZE[1]
    w_size = LAG_WSIZE[2]
    method: str = LAG_WSIZE[3]
    col_name = f"fe_rolling_{method}_t{lag}_{w_size}"

    with timer("create {}".format(col_name)):
        if method == "mean":
            data[col_name] = (
                data.groupby(["id"])["sales"]
                .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                .astype(np.float16)
            )
        if method == "std":
            data[col_name] = (
                data.groupby(["id"])["sales"]
                .transform(lambda x: x.shift(lag).rolling(w_size).std())
                .astype(np.float16)
            )
        if method == "sum":
            data[col_name] = (
                data.groupby(["id"])["sales"]
                .transform(lambda x: x.shift(lag).rolling(w_size).sum())
                .astype(np.float16)
            )
    return data[[col_name]]


def make_rolling_for_test(
    test: pd.DataFrame, d: int, features: List[str]
) -> pd.DataFrame:
    for lag in range(1, 28):
        if f"shift_t{lag}" in features:
            test.loc[test.d == d, f"shift_t{lag}"] = test.loc[
                test.d == (d - lag), "sales"
            ].values
    for lag in [1, 7, 14]:
        for w_size in [7, 14, 30, 60, 90, 180]:
            for method in ["mean", "std"]:
                col_name = f"fe_rolling_{method}_t{lag}_{w_size}"
                if col_name in features:
                    group = test[
                        (test.d >= d - lag - w_size) & (test.d <= d - lag)
                    ].groupby("id")
                    test.loc[test.d == d, col_name] = (
                        group.agg({"sales": method})
                        .reindex(test.loc[test.d == d, "id"])["sales"]
                        .values
                    )
    return test


class FERollingSum(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling sum"):
            lag_wsize = []
            for lag in [0, 1]:
                for w_size in [30, 50]:
                    lag_wsize.append([data[["id", "d", "sales"]], lag, w_size, "sum"])
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
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
            lag_wsize = []
            for lag in [1, 7, 14, 28]:
                for w_size in [7, 14, 30, 60, 90, 180]:
                    lag_wsize.append([data[["id", "d", "sales"]], lag, w_size, "mean"])
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_mean")
        print(df.info())
        self.dump(df)


class FERollingStd(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling std"):
            lag_wsize = []
            for lag in [1, 7, 14, 28]:
                for w_size in [7, 30, 60, 90]:
                    lag_wsize.append([data[["id", "d", "sales"]], lag, w_size, "std"])
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
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
