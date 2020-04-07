from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np


class FETarget(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("target encoding"):
            data["fe_target_mean"] = data.groupby(["id", "tm_dw"])["sales"].transform(
                lambda x: np.nanmean(x)
            )
            data["fe_target_std"] = data.groupby(["id", "tm_dw"])["sales"].transform(
                lambda x: np.nanstd(x)
            )
            data["fe_target_max"] = data.groupby(["id", "tm_dw"])["sales"].transform(
                lambda x: np.nanmax(x)
            )
        df = data.filter(like="fe_target_")
        print_mem_usage(df)
        self.dump(df)
