import gc
import pandas as pd
import numpy as np
from typing import List, Dict
import sklearn.preprocessing
from kaggle_m5_forecasting import M5, LoadRawData, RawData
from kaggle_m5_forecasting.utils import (
    timer,
    reduce_mem_usage,
    print_mem_usage,
    merge_by_concat,
)
import pickle
from tqdm import tqdm
import math
import decimal
from datetime import datetime


def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
    dec = decimal.Decimal
    diff = d - datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    phase_index = math.floor((lunations % dec(1) * dec(8)) + dec("0.5"))
    return int(phase_index)


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
                tmp_df["d"] = f"d_{1941+i}"
                tmp_df["sales"] = np.nan
                add_df = pd.concat([add_df, tmp_df])
            data = pd.concat([data, add_df]).reset_index(drop=True)
            del add_df
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

        with timer("make calendar events"):
            raw.calendar["cal_christmas_eve"] = (
                raw.calendar["date"].str[5:] == "12-24"
            ).astype(np.int8)
            raw.calendar["cal_christmas"] = (
                raw.calendar["date"].str[5:] == "12-25"
            ).astype(np.int8)
            raw.calendar["cal_blackfriday"] = (
                raw.calendar["date"]
                .str[5:]
                .isin(
                    [
                        "2011-11-25",
                        "2012-11-23",
                        "2013-11-29",
                        "2014-11-28",
                        "2015-11-27",
                    ]
                )
            ).astype(np.int8)
            raw.calendar.loc[
                raw.calendar["cal_blackfriday"] == 1, "event_name_1"
            ] = "BlackFriday"
            raw.calendar.loc[
                raw.calendar["cal_blackfriday"] == 1, "event_type_1"
            ] = "other"
            raw.calendar["event_name_1_yesterday"] = raw.calendar["event_name_1"].shift(
                1
            )
            raw.calendar["event_type_1_yesterday"] = raw.calendar["event_type_1"].shift(
                1
            )
            raw.calendar["event_name_1_tomorrow"] = raw.calendar["event_name_1"].shift(
                -1
            )
            raw.calendar["event_type_1_tomorrow"] = raw.calendar["event_type_1"].shift(
                -1
            )

        with timer("merge calendar"):
            icols = [
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "event_name_1_tomorrow",
                "event_type_1_tomorrow",
                "event_name_1_yesterday",
                "event_type_1_yesterday",
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

        with timer("make snap"):
            data["snap"] = 0
            data.loc[(data.snap_CA == 1) & (data.state_id == "CA"), "snap"] = 1
            data.loc[(data.snap_TX == 1) & (data.state_id == "TX"), "snap"] = 1
            data.loc[(data.snap_WI == 1) & (data.state_id == "WI"), "snap"] = 1

        with timer("make some features from date"):
            data["tm_d"] = data["date"].dt.day.astype(np.int8)
            data["tm_w"] = data["date"].dt.week.astype(np.int8)
            data["tm_m"] = data["date"].dt.month.astype(np.int8)
            data["tm_y"] = data["date"].dt.year
            data["tm_quarter"] = data["date"].dt.quarter.astype(np.int8)
            data["tm_y"] = (data["tm_y"] - data["tm_y"].min()).astype(np.int8)
            data["tm_wm"] = data["tm_d"].apply(lambda x: np.ceil(x / 7)).astype(np.int8)
            data["tm_dw"] = data["date"].dt.dayofweek.astype(np.int8)
            data["tm_w_end"] = (data["tm_dw"] >= 5).astype(np.int8)
            data["tm_moon_phase"] = (
                data["date"].map(lambda d: get_moon_phase(d)).astype(np.int8)
            )
            data.loc[data["event_type_1"] == "National", "tm_w_end"] = 1
            del data["date"]
            print_mem_usage(data)

        with timer("merge sell_prices"):
            data = data.merge(
                raw.sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
            )

        with timer("convert 'd' to int"):
            data["d"] = data["d"].apply(lambda x: x[2:]).astype(np.int16)
            data["sales_is_zero"] = (data["sales"] == 0).astype(np.int8)
            print_mem_usage(data)

        with timer("label encoding"):
            cat_encoders: Dict[str, sklearn.preprocessing.LabelEncoder] = {}
            cat_features: List[str] = [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "event_name_1_yesterday",
                "event_type_1_yesterday",
                "event_name_1_tomorrow",
                "event_type_1_tomorrow",
            ]
            for feature in tqdm(cat_features):
                encoder = sklearn.preprocessing.LabelEncoder()
                encoder.fit(data[feature])
                data[feature] = encoder.transform(data[feature])
                cat_encoders[feature] = encoder
            pickle.dump(cat_encoders, open("./cat_encoders.pkl", "wb"))

        print(data.info())

        self.dump(data)
