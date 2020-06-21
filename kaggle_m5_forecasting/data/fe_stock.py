import pandas as pd
from typing import Dict
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.load_data import LoadRawData, RawData
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.utils import timer


class FEStock(M5):
    def requires(self):
        return dict(raw=LoadRawData(), data=MakeData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")

        raw.calendar["d"] = raw.calendar["d"].map(lambda d: int(d.replace("d_", "")))

        stock = pd.read_csv("./external_data/stock.csv")
        stock.columns = ["date", "close_last", "volume", "open", "high", "low"]
        stock["date"] = pd.to_datetime(stock["date"]).dt.strftime("%Y-%m-%d")
        for col in ["close_last", "open", "high", "low"]:
            stock[col] = stock[col].map(lambda x: float(x.replace("$", "")))
        stock = stock[["date", "close_last", "volume"]]
        stock.columns = ["date", "fe_stock_price", "fe_stock_volume"]
        stock = raw.calendar[["date"]].merge(stock, on="date", how="left")
        stock["fe_stock_price"] = (
            stock["fe_stock_price"].fillna(method="ffill").fillna(method="bfill")
        )
        stock["fe_stock_volume"] = (
            stock["fe_stock_volume"].fillna(method="ffill").fillna(method="bfill")
        )

        with timer("merge data"):
            data = data.merge(raw.calendar[["d", "date"]], on="d", how="left").merge(
                stock, on="date", how="left"
            )

        df = data.filter(like="fe_stock")
        print(df.info())

        self.dump(df)
