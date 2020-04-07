from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np


class FEPriceRolling(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        with timer("make rolling_price_std_t7"):
            data["fe_rolling_price_std_t7"] = (
                data.groupby(["id"])["sell_price"]
                .transform(lambda x: x.rolling(7).std())
                .astype(np.float16)
            )

        with timer("make rolling_price_std_t30"):
            data["fe_rolling_price_std_t30"] = (
                data.groupby(["id"])["sell_price"]
                .transform(lambda x: x.rolling(30).std())
                .astype(np.float16)
            )

        df = data.filter(like="fe_rolling_price")
        print(df.info())
        self.dump(df)
