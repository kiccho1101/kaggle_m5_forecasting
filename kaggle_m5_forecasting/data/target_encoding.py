from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.base import SplitIndex
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.data.split_data import SplitValData, SplitData
from kaggle_m5_forecasting.utils import timer

import pandas as pd
from tqdm.autonotebook import tqdm


def target_encoding(data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    groups = [["item_id"], ["cat_id"], ["dept_id"], ["item_id", "tm_dw"]]
    methods = ["mean", "std"]

    with timer("target encoding"):
        for group in tqdm(groups):
            tmp_df = train_df.groupby(["store_id"] + group).agg({"sales": methods})
            tmp_df.columns = [
                "fe_te_{}_{}".format("_".join(group), method) for method in methods
            ]
            tmp_df.reset_index(inplace=True)
            data = data.merge(tmp_df, on=["store_id"] + group, how="left")
    return data


class TEValData(M5):
    def requires(self):
        return dict(data=MakeData(), sp_idx=SplitValData())

    def run(self):
        data: pd.DataFrame = self.load("data")
        sp_idx: SplitIndex = self.load("sp_idx")
        train_df: pd.DataFrame = data.iloc[sp_idx.train, :]
        data = target_encoding(data, train_df)
        df = data.filter(like="fe_te_")
        print(df.info())
        self.dump(df)


class TEData(M5):
    def requires(self):
        return dict(data=MakeData(), sp_idx=SplitData())

    def run(self):
        data: pd.DataFrame = self.load("data")
        sp_idx: SplitIndex = self.load("sp_idx")
        train_df: pd.DataFrame = data.iloc[sp_idx.train, :]
        data = target_encoding(data, train_df)
        df = data.filter(like="fe_te_")
        print(df.info())
        self.dump(df)
