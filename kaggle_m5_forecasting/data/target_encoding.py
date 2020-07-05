from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.base import SplitIndex
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.data.split_data import SplitValData, SplitData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage
from kaggle_m5_forecasting.data.fe_event import FEEvent

from typing import List, Tuple
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from kaggle_m5_forecasting.config import Config


def target_encoding(train_df: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        ["item_id"],
        ["item_id", "tm_w"],
        ["item_id", "tm_dw"],
        ["dept_id", "tm_w"],
        ["cat_id", "tm_w"],
        ["store_id", "dept_id"],
        ["store_id", "dept_id", "tm_w"],
        ["store_id", "dept_id", "tm_m"],
        ["store_id", "tm_w"],
        ["store_id", "tm_m"],
        ["store_id", "tm_d"],
        ["store_id", "snap"],
        ["store_id", "snap", "tm_dw"],
        ["state_id", "item_id"],
        ["state_id", "item_id", "tm_dw"],
        ["state_id", "item_id", "tm_w"],
        ["state_id", "item_id", "tm_m"],
        ["state_id", "item_id", "snap"],
        ["state_id", "item_id", "snap", "tm_dw"],
        ["state_id", "item_id", "fe_event"],
        ["state_id", "item_id", "fe_event_dw"],
        ["store_id", "item_id"],
        ["store_id", "item_id", "tm_dw"],
        ["store_id", "item_id", "tm_w"],
        ["store_id", "item_id", "tm_m"],
        ["store_id", "item_id", "tm_d"],
        ["store_id", "item_id", "snap"],
        ["store_id", "item_id", "snap", "tm_dw"],
        ["store_id", "item_id", "fe_event"],
        ["store_id", "item_id", "fe_event_dw"],
    ]

    result: List[Tuple[List[str], pd.DataFrame]] = []
    methods = ["mean", "std"]
    with timer("target encoding"):
        for group_key in tqdm(group_keys):
            columns = []
            columns += group_key
            columns += [
                "fe_te_{}_{}".format("_".join(group_key), method) for method in methods
            ]
            tmp_df = (
                train_df[group_key + ["sales"]]
                .groupby(group_key)
                .agg({"sales": methods})
                .reset_index()
            )
            tmp_df.columns = columns
            tmp_df.reset_index(inplace=True, drop=True)
            result.append((group_key, tmp_df))
    return result


class TEValData(M5):
    def requires(self):
        return dict(data=MakeData(), fe_event=FEEvent())

    def run(self):
        config = Config()
        data: pd.DataFrame = pd.concat(
            [self.load("data"), self.load("fe_event")], axis=1
        )
        dfs: List[pd.DataFrame] = []
        for end_day in config.CV_START_DAYS:
            with timer("create grouped df"):
                # train_df: pd.DataFrame = data[
                #     (data.d > config.START_DAY) & (data.d < end_day)
                # ]
                train_df: pd.DataFrame = data[data.d < end_day]
                grouped: List[Tuple[List[str], pd.DataFrame]] = target_encoding(
                    train_df
                )
            with timer("merge into data"):
                df = data.copy()
                for group_key, grouped_df in tqdm(grouped):
                    df = df.merge(grouped_df, on=group_key, how="left")
                df = reduce_mem_usage(df.filter(like="fe_te_"))
                print(df.info())
                dfs.append(df)
        self.dump(dfs)


class TEData(M5):
    def requires(self):
        return dict(data=MakeData(), fe_event=FEEvent())

    def run(self):
        config = Config()
        data: pd.DataFrame = pd.concat(
            [self.load("data"), self.load("fe_event")], axis=1
        )
        train_df: pd.DataFrame = data[(data.d > config.START_DAY) & (data.d <= 1913)]
        # train_df = train_df.sample(int(len(train_df) * 0.15))
        with timer("create grouped df"):
            grouped: List[Tuple[List[str], pd.DataFrame]] = target_encoding(train_df)
        with timer("merge into data"):
            for group_key, grouped_df in tqdm(grouped):
                data = data.merge(grouped_df, on=group_key, how="left")
            df = reduce_mem_usage(data.filter(like="fe_te_"))
            print(df.info())
        self.dump(df)
