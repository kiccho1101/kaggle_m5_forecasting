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
    df: pd.DataFrame = LAG_WSIZE[0]
    lag = LAG_WSIZE[1]
    w_size = LAG_WSIZE[2]
    method: str = LAG_WSIZE[3]
    col_name = f"fe_rolling_{method}_t{lag}_{w_size}"

    group = df.groupby("id")
    with timer("create {}".format(col_name)):
        if method == "mean":
            df[col_name] = (
                group["sales"]
                .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                .astype(np.float16)
            )
        if method == "ewm_mean":
            df[col_name] = (
                group["sales"]
                .transform(lambda x: x.shift(lag).ewm(alpha=0.03).mean())
                .astype(np.float16)
            )
        if method == "std":
            df[col_name] = (
                group["sales"]
                .transform(lambda x: x.shift(lag).rolling(w_size).std())
                .astype(np.float16)
            )
        if method == "sum":
            df[col_name] = (
                group["sales"]
                .transform(lambda x: x.shift(lag).rolling(w_size).sum())
                .astype(np.float16)
            )
        if method == "zero_ratio":
            df[col_name] = (
                group["sales_is_zero"]
                .transform(lambda x: x.shift(lag).rolling(w_size).mean())
                .astype(np.float16)
            )
        if method == "percentile25":
            df[col_name] = group["sales"].transform(
                lambda x: x.shift(lag)
                .rolling(w_size)
                .apply(lambda y: np.sort(y)[: int(len(y) * 0.25)].mean())
            )
        if method == "percentile50":
            df[col_name] = group["sales"].transform(
                lambda x: x.shift(lag)
                .rolling(w_size)
                .apply(lambda y: np.sort(y)[: int(len(y) * 0.5)].mean())
            )
        if method == "percentile75":
            df[col_name] = group["sales"].transform(
                lambda x: x.shift(lag)
                .rolling(w_size)
                .apply(lambda y: np.sort(y)[: int(len(y) * 0.75)].mean())
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
    for lag in [7]:
        for method in ["ewm_mean"]:
            col_name = f"fe_rolling_{method}_t{lag}_0"
            if col_name in features:
                test[col_name] = test.groupby("id")["sales"].apply(
                    lambda x: x.ewm(alpha=0.03).mean()
                )
    for lag in [7]:
        for w_size in [7, 30, 60, 90, 180]:
            for method in [
                "mean",
                "std",
                "skew",
                "zero_ratio",
                "percentile25",
                "percentile50",
                "percentile75",
            ]:
                col_name = f"fe_rolling_{method}_t{lag}_{w_size}"
                if col_name in features:
                    group = test[
                        (test.d >= d - lag - w_size) & (test.d <= d - lag)
                    ].groupby("id")
                    if method == "zero_ratio":
                        test.loc[test.d == d, col_name] = (
                            group.agg({"sales_is_zero": "mean"})
                            .reindex(test.loc[test.d == d, "id"])["sales_is_zero"]
                            .values
                        )
                    elif method == "percentile25":
                        test.loc[test.d == d, col_name] = (
                            group.agg(
                                {
                                    "sales": lambda x: np.sort(x)[
                                        : int(len(x) * 0.25)
                                    ].mean()
                                }
                            )
                            .reindex(test.loc[test.d == d, "id"])["sales"]
                            .values
                        )
                    elif method == "percentile50":
                        test.loc[test.d == d, col_name] = (
                            group.agg(
                                {
                                    "sales": lambda x: np.sort(x)[
                                        : int(len(x) * 0.50)
                                    ].mean()
                                }
                            )
                            .reindex(test.loc[test.d == d, "id"])["sales"]
                            .values
                        )
                    elif method == "percentile75":
                        test.loc[test.d == d, col_name] = (
                            group.agg(
                                {
                                    "sales": lambda x: np.sort(x)[
                                        : int(len(x) * 0.75)
                                    ].mean()
                                }
                            )
                            .reindex(test.loc[test.d == d, "id"])["sales"]
                            .values
                        )
                    else:
                        test.loc[test.d == d, col_name] = (
                            group.agg({"sales": method})
                            .reindex(test.loc[test.d == d, "id"])["sales"]
                            .values
                        )
                diff_col_name = f"fe_rolling_{method}_diff_t{lag}_{w_size}"
                if diff_col_name in features:
                    test.loc[(test.d == d), diff_col_name] = (
                        test.loc[(test.d == d), col_name].values
                        - test.loc[(test.d == d - lag), "sales"].values
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
            for lag in [7, 28]:
                for w_size in [7, 30, 60, 90, 180]:
                    lag_wsize.append([data[["id", "d", "sales"]], lag, w_size, "mean"])
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_mean")
        print(df.info())
        self.dump(df)


class FERollingEwmMean(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling ewm_mean"):
            lag_wsize = []
            for lag in [7, 28]:
                for w_size in [0]:
                    lag_wsize.append(
                        [data[["id", "d", "sales"]], lag, w_size, "ewm_mean"]
                    )
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_ewm_mean")
        print(df.info())
        self.dump(df)


class FERollingZeroRatio(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling zero_ratio"):
            lag_wsize = []
            for lag in [7, 28]:
                for w_size in [7, 30, 60, 90, 180]:
                    lag_wsize.append(
                        [
                            data[["id", "d", "sales", "sales_is_zero"]],
                            lag,
                            w_size,
                            "zero_ratio",
                        ]
                    )
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_zero_ratio")
        print(df.info())
        self.dump(df)


class FERollingPercentile25(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling percentile25"):
            lag_wsize = []
            for lag in [7, 28]:
                for w_size in [30, 60, 90, 180]:
                    lag_wsize.append(
                        [data[["id", "d", "sales"]], lag, w_size, "percentile25"]
                    )
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_percentile25")
        print(df.info())
        self.dump(df)


class FERollingPercentile50(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling percentile50"):
            lag_wsize = []
            for lag in [7, 28]:
                for w_size in [30, 60, 90, 180]:
                    lag_wsize.append(
                        [data[["id", "d", "sales"]], lag, w_size, "percentile50"]
                    )
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_percentile50")
        print(df.info())
        self.dump(df)


class FERollingPercentile75(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling percentile75"):
            lag_wsize = []
            for lag in [7, 28]:
                for w_size in [30, 60, 90, 180]:
                    lag_wsize.append(
                        [data[["id", "d", "sales"]], lag, w_size, "percentile75"]
                    )
            data = pd.concat(
                [data, df_parallelize_run(make_lag_roll, lag_wsize)], axis=1
            )
        df = data.filter(like="fe_rolling_percentile75")
        print(df.info())
        self.dump(df)


class FERollingMeanDiff(M5):
    def requires(self):
        return dict(data=MakeData(), roll=FERollingMean())

    def run(self):
        data: pd.DataFrame = self.load("data")
        roll: pd.DataFrame = self.load("roll")

        # %%
        for col in roll.columns:
            lag = int(col.split("_")[3].replace("t", ""))
            data[col.replace("fe_rolling_mean", "fe_rolling_mean_diff")] = roll[
                col
            ] - data["sales"].shift(lag)
        df = data.filter(like="fe_rolling_mean_diff")
        print(df.info())
        self.dump(df)


class FERollingStd(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make rolling std"):
            lag_wsize = []
            for lag in [7, 28]:
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
