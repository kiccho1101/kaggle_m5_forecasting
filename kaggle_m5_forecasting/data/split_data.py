import pandas as pd
import numpy as np
import random
from kaggle_m5_forecasting import M5, config, MakeData
from kaggle_m5_forecasting.base import SplitIndex


class SplitValData(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        np.random.seed(config.SEED)
        random.seed(config.SEED)
        sp_idx: SplitIndex = SplitIndex()
        train_df = data[(data.d > config.START_DAY) & (data.d <= 1885)]
        sp_idx.train = list(train_df.sample(int(len(train_df) * 0.05)).index)
        sp_idx.test = list(
            data[(data.d > 1885 - config.MAX_LAGS) & (data.d <= 1913)].index
        )

        print("train shape:", data.iloc[sp_idx.train, :].shape)
        print("test shape:", data.iloc[sp_idx.test, :].shape)

        self.dump(sp_idx)


class SplitData(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        sp_idx: SplitIndex = SplitIndex()

        sp_idx.train = list(data[(data.d >= config.START_DAY) & (data.d <= 1913)].index)
        sp_idx.test = list(data[(data.d > 1913 - config.MAX_LAGS)].index)

        print("train shape:", data.iloc[sp_idx.train, :].shape)
        print("test shape:", data.iloc[sp_idx.test, :].shape)

        self.dump(sp_idx)
