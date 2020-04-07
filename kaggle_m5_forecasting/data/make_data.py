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

        with timer("merge sell_prices"):
            data = data.merge(
                raw.sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
            )

        with timer("convert 'd' to int"):
            data["d"] = data["d"].apply(lambda x: x[2:]).astype(np.int16)
            print_mem_usage(data)

        print(data.info())

        self.dump(data)