import gc
import pandas as pd
import numpy as np
from kaggle_m5_forecasting import M5, LoadRawData, RawData
from kaggle_m5_forecasting.utils import (
    timer,
    reduce_mem_usage,
    print_mem_usage,
    merge_by_concat,
)
from tqdm import tqdm


class MakeData(M5):
    def requires(self):
        return LoadRawData()

    def run(self):
        raw: RawData = self.load()

        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        with timer("melt sales_train_validation"):
            data: pd.DataFrame = pd.melt(
                raw.sales_train_validation,
                id_vars=id_vars,
                var_name="d",
                value_name="sales",
            )
            print_mem_usage(data)

        with timer("add test data"):
            add_df = pd.DataFrame()
            for i in tqdm(range(1, 29)):
                tmp_df = raw.sales_train_validation[id_vars].drop_duplicates()
                tmp_df["d"] = f"d_{1913+i}"
                tmp_df["sales"] = np.nan
                add_df = pd.concat([add_df, tmp_df])
            data = pd.concat([data, add_df]).reset_index(drop=True)
            del tmp_df, add_df
            print_mem_usage(data)

        with timer("str to category"):
            for col in tqdm(id_vars):
                data[col] = data[col].astype("category")
            print_mem_usage(data)

        with timer("merge release"):
            data = merge_by_concat(
                data,
                raw.sell_prices.groupby(["store_id", "item_id"])["wm_yr_wk"]
                .agg(release=np.min)
                .reset_index(),
                ["store_id", "item_id"],
            )
            print_mem_usage(data)

        with timer("merge wm_yr_wk"):
            data = merge_by_concat(data, raw.calendar[["wm_yr_wk", "d"]], ["d"])
            print_mem_usage(data)

        with timer("cutoff data before release"):
            data = data[data["wm_yr_wk"] >= data["release"]].reset_index(drop=True)
            print_mem_usage(data)

        prices_df: pd.DataFrame = raw.sell_prices.copy()
        with timer("basic price aggregations"):
            prices_df["price_max"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.max)
            prices_df["price_min"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.min)
            prices_df["price_std"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.std)
            prices_df["price_mean"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.mean)
            prices_df["price_norm"] = prices_df["sell_price"] / prices_df["price_max"]
            prices_df["price_nunique"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform("nunique")
            prices_df["item_nunique"] = prices_df.groupby(["store_id", "sell_price"])[
                "item_id"
            ].transform("nunique")
            prices_df = prices_df.merge(
                raw.calendar[["wm_yr_wk", "month", "year"]].drop_duplicates(
                    subset=["wm_yr_wk"]
                ),
                on=["wm_yr_wk"],
                how="left",
            )

        with timer("calc price momentum"):
            prices_df["price_momentum"] = prices_df["sell_price"] / prices_df.groupby(
                ["store_id", "item_id"]
            )["sell_price"].transform(lambda x: x.shift(1))
            prices_df["price_momentum_m"] = prices_df["sell_price"] / prices_df.groupby(
                ["store_id", "item_id", "month"]
            )["sell_price"].transform("mean")
            prices_df["price_momentum_y"] = prices_df["sell_price"] / prices_df.groupby(
                ["store_id", "item_id", "year"]
            )["sell_price"].transform("mean")
            del prices_df["month"], prices_df["year"]

        with timer("merge prices_df"):
            data = data.merge(
                prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left"
            )
            print_mem_usage(data)

        reduce_mem_usage(data)

        with timer("merge calendar"):
            icols = [
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
            data = data.merge(
                raw.calendar.drop(
                    ["wm_yr_wk", "weekday", "wday", "month", "year"], axis=1
                ),
                on=["d"],
                how="left",
            )
            for col in tqdm(icols):
                data[col].fillna("unknown", inplace=True)
                data[col] = data[col].astype("category")
            data["date"] = pd.to_datetime(data["date"])
            print_mem_usage(data)

        with timer("make some features from date"):
            data["tm_d"] = data["date"].dt.day.astype(np.int8)
            data["tm_w"] = data["date"].dt.week.astype(np.int8)
            data["tm_m"] = data["date"].dt.month.astype(np.int8)
            data["tm_y"] = data["date"].dt.year
            data["tm_y"] = (data["tm_y"] - data["tm_y"].min()).astype(np.int8)
            data["tm_wm"] = data["tm_d"].apply(lambda x: np.ceil(x / 7)).astype(np.int8)

            data["tm_dw"] = data["date"].dt.dayofweek.astype(np.int8)
            data["tm_w_end"] = (data["tm_dw"] >= 5).astype(np.int8)
            del data["date"]
            print_mem_usage(data)

        with timer("convert 'd' to int"):
            data["d"] = data["d"].apply(lambda x: x[2:]).astype(np.int16)
            print_mem_usage(data)

        print(data.info())

        self.dump(data)
