import pandas as pd
import numpy as np
import random
from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.config import Config
from kaggle_m5_forecasting.base import SplitIndex
from typing import List

from tqdm.autonotebook import tqdm


class SplitValData(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        sp_idxs: List[SplitIndex] = []
        config = Config()
        np.random.seed(config.SEED)
        random.seed(config.SEED)

        for cv_start_day in tqdm(config.CV_START_DAYS):
            sp_idx: SplitIndex = SplitIndex()
            train_df = data[data.d < cv_start_day]
            sp_idx.train = list(
                # train_df.sample(int(len(data) * config.CV_SAMPLE_RATE)).index
                train_df.index
            )
            sp_idx.test = list(
                data[
                    (data.d >= cv_start_day - config.MAX_LAGS)
                    & (data.d < cv_start_day + 28)
                ].index
            )
            sp_idxs.append(sp_idx)

        self.dump(sp_idxs)


class SplitData(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        sp_idx: SplitIndex = SplitIndex()

        config = Config()
        sp_idx.train = list(data[(data.d >= config.START_DAY) & (data.d <= 1913)].index)
        sp_idx.test = list(data[(data.d > 1913 - config.MAX_LAGS)].index)

        print("train shape:", data.iloc[sp_idx.train, :].shape)
        print("test shape:", data.iloc[sp_idx.test, :].shape)

        self.dump(sp_idx)
