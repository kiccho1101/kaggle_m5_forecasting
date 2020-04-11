import pandas as pd
import numpy as np
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.load_data import LoadRawData, RawData
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.utils import (
    timer,
    reduce_mem_usage,
    print_mem_usage,
    merge_by_concat,
)
from tqdm import tqdm


class FEPriceBasic(M5):
    def requires(self):
        return dict(data=MakeData(), raw=LoadRawData())

    def run(self):
        data: pd.DataFrame = self.load("data")
        raw: RawData = self.load("raw")

        prices_df: pd.DataFrame = raw.sell_prices.copy()
        with timer("basic price aggregations"):
            prices_df["fe_price_max"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.max)
            prices_df["fe_price_min"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.min)
            prices_df["fe_price_std"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.std)
            prices_df["fe_price_mean"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(np.mean)
            prices_df["fe_price_discount"] = (
                prices_df["fe_price_mean"] - prices_df["sell_price"]
            )
            prices_df["fe_price_discount_rate"] = (
                prices_df["fe_price_discount"] / prices_df["fe_price_mean"]
            )
            prices_df["fe_price_skew"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(lambda x: x.skew())
            prices_df["fe_price_kurt"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform(lambda x: x.kurt())
            prices_df["fe_price_norm"] = (
                prices_df["sell_price"] / prices_df["fe_price_max"]
            )
            prices_df["fe_price_nunique"] = prices_df.groupby(["store_id", "item_id"])[
                "sell_price"
            ].transform("nunique")
            prices_df["fe_price_item_nunique"] = prices_df.groupby(
                ["store_id", "sell_price"]
            )["item_id"].transform("nunique")
            prices_df = prices_df.merge(
                raw.calendar[["wm_yr_wk", "month", "year"]].drop_duplicates(
                    subset=["wm_yr_wk"]
                ),
                on=["wm_yr_wk"],
                how="left",
            )

        with timer("calc price momentum"):
            prices_df["fe_price_momentum"] = prices_df[
                "sell_price"
            ] / prices_df.groupby(["store_id", "item_id"])["sell_price"].transform(
                lambda x: x.shift(1)
            )
            prices_df["fe_price_momentum_m"] = prices_df[
                "sell_price"
            ] / prices_df.groupby(["store_id", "item_id", "month"])[
                "sell_price"
            ].transform(
                "mean"
            )
            prices_df["fe_price_momentum_y"] = prices_df[
                "sell_price"
            ] / prices_df.groupby(["store_id", "item_id", "year"])[
                "sell_price"
            ].transform(
                "mean"
            )
            del prices_df["month"], prices_df["year"]

        with timer("merge prices_df"):
            data = data.merge(
                prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left"
            )

        df = data.filter(like="fe_price")
        df = reduce_mem_usage(df)
        print(df.info())
        self.dump(df)
