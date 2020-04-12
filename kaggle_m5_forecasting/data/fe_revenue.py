import pandas as pd
import numpy as np
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.utils import (
    timer,
    reduce_mem_usage,
    print_mem_usage,
    merge_by_concat,
)
from tqdm import tqdm


class FERevenue(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        data["fe_revenue"] = data["sales"] * data["sell_price"]
        data["fe_revenue_dept_sum"] = data.groupby(["store_id", "dept_id", "d"])[
            "fe_revenue"
        ].transform(np.nansum)

        df = data.filter(like="fe_revenue")
        df = reduce_mem_usage(df)
        print(df.info())
        self.dump(df)
