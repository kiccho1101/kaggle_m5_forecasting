from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import (
    timer,
    df_parallelize_run,
)
from typing import List, Any
from tqdm import tqdm
import pandas as pd
import numpy as np


def make_lag_roll(LAG_WSIZE: List[Any]):
    df: pd.DataFrame = LAG_WSIZE[0]
    lag = LAG_WSIZE[1]
    w_size = LAG_WSIZE[2]
    method: str = LAG_WSIZE[3]
    # group_ids: List[str] = df.drop(["id", "d", "sales"]).columns.tolist()
    print(lag, w_size, method)

    col_name: str = ""
    if method == "group_mean":
        pass
        # col_name = "fe_rolling_{}_mean_{}_{}".format("_".join(group_ids), lag, w_size)
        # with timer("create {}".format(col_name)):
        #     _tmp = df.groupby(["d"] + group_ids)["sales"].mean().reset_index()
        #     _tmp[col_name] = _tmp.groupby(group_ids)["sales"].transform(
        #         lambda x: x.shift(lag).rolling(w_size).mean()
        #     )
        #     _tmp.drop("sales", axis=1, inplace=True)
        #     df = df.merge(_tmp, on=["d"] + group_ids, how="left")

    else:
        col_name = f"fe_rolling_{method}_t{lag}_{w_size}"
        with timer("create {}".format(col_name)):
            if method == "mean":
                df[col_name] = (
                    df.groupby("id")["sales"]
                    .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                    .astype(np.float16)
                )
            if method == "std":
                df[col_name] = (
                    df.groupby("id")["sales"]["sales"]
                    .transform(lambda x: x.shift(lag).rolling(w_size).std())
                    .astype(np.float16)
                )
            if method == "dw_mean":
                df[col_name] = (
                    df.groupby(["id", "tm_dw"])["sales"]
                    .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                    .astype(np.float16)
                )

    return df[[col_name]]


def make_rolling_for_test(
    test: pd.DataFrame, d: int, features: List[str]
) -> pd.DataFrame:
    for lag in range(1, 28):
        if f"shift_t{lag}" in features:
            test.loc[test.d == d, f"shift_t{lag}"] = test.loc[
                test.d == (d - lag), "sales"
            ].values
    for lag in [1, 7]:
        for w_size in [7, 14, 30, 60, 90, 180]:
            for method in [
                "mean",
                "std",
                "skew",
            ]:
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


class FERollingMean(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling mean"):
            lag_wsize = []
            for lag in [1, 14, 7, 28]:
                for w_size in [7, 30, 60, 90, 180]:
                    lag_wsize.append([data[["id", "d", "sales"]], lag, w_size, "mean"])
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_mean")
        print(df.info())
        self.dump(df)


class FERollingDWMean(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling mean"):
            lag_wsize = []
            for lag in [4]:
                for w_size in [1, 4, 8, 16]:
                    lag_wsize.append(
                        [data[["id", "d", "tm_dw", "sales"]], lag, w_size, "dw_mean"]
                    )
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_dw_mean")
        print(df.info())

        self.dump(df)


class FERollingGroupMean(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        groups = [
            ["item_id"],
            ["store_id"],
            ["state_id", "item_id"],
        ]
        with timer("make rolling cat_id, item_id mean"):
            # lag_wsize = []
            for group in tqdm(groups):
                for lag in tqdm([28]):
                    for w_size in tqdm([7, 30, 180]):
                        col_name = "fe_rolling_{}_mean_{}_{}".format(
                            "_".join(group), lag, w_size
                        )
                        _tmp = data.groupby(["d"] + group)["sales"].mean().reset_index()
                        _tmp[col_name] = _tmp.groupby(group)["sales"].transform(
                            lambda x: x.shift(lag).rolling(w_size).mean()
                        )
                        _tmp.drop("sales", axis=1, inplace=True)
                        data = data.merge(_tmp, on=["d"] + group, how="left")
        df = data.filter(like="fe_rolling_")
        print(df.info())
        self.dump(df)


class FERollingGroupStd(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        groups = [
            ["item_id"],
            ["store_id"],
            ["state_id", "item_id"],
        ]
        with timer("make group std"):
            for group in tqdm(groups):
                for lag in tqdm([28]):
                    for w_size in tqdm([7, 30, 180]):
                        col_name = "fe_rolling_{}_std_{}_{}".format(
                            "_".join(group), lag, w_size
                        )
                        _tmp = data.groupby(["d"] + group)["sales"].mean().reset_index()
                        _tmp[col_name] = _tmp.groupby(group)["sales"].transform(
                            lambda x: x.shift(lag).rolling(w_size).std()
                        )
                        _tmp.drop("sales", axis=1, inplace=True)
                        data = data.merge(_tmp, on=["d"] + group, how="left")
        df = data.filter(like="fe_rolling_")
        print(df.info())
        self.dump(df)


class FERollingStd(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling std"):
            lag_wsize = []
            for lag in [1, 7, 28]:
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
