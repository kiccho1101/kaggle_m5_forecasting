from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.base import SplitIndex
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.data.split_data import SplitValData, SplitData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage

from typing import List, Tuple
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from kaggle_m5_forecasting.config import Config


def target_encoding(train_df: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        ["item_id"],
        ["item_id", "tm_dw"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
        ["store_id", "dept_id", "tm_dw"],
        ["store_id", "item_id"],
        ["store_id", "item_id", "tm_dw"],
        # ["store_id", "tm_season_t8"],
        # ["store_id", "item_id", "tm_season_t5"],
        # ["store_id", "tm_dw", "tm_season_t5"],
        # ["store_id", "item_id", "tm_dw", "tm_season_t5"],
        # ["store_id", "item_id", "tm_season_t8"],
        # ["store_id", "item_id", "tm_season_t10"],
    ]
    result: List[Tuple[List[str], pd.DataFrame]] = []
    methods = ["mean", "std", "skew"]
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
        return dict(data=MakeData())

    def run(self):
        config = Config()
        data: pd.DataFrame = self.load("data")
        dfs: List[pd.DataFrame] = []
        for end_day in config.CV_START_DAYS:
            with timer("create grouped df"):
                train_df: pd.DataFrame = data[
                    (data.d > config.START_DAY) & (data.d < end_day)
                ]
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
        return MakeData()

    def run(self):
        config = Config()
        data: pd.DataFrame = self.load()
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
