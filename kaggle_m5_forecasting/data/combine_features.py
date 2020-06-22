from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.base import Split, SplitIndex
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.data.fe_price_basic import FEPriceBasic
from kaggle_m5_forecasting.data.fe_price_change import FEPriceChange
from kaggle_m5_forecasting.data.fe_price_rolling import FEPriceRolling
from kaggle_m5_forecasting.data.fe_shift import FEShift
from kaggle_m5_forecasting.data.fe_rolling import (
    FERollingMean,
    FERollingGroupMean,
    FERollingGroupStd,
    FERollingStd,
    FERollingKurt,
    FERollingSkew,
)
from kaggle_m5_forecasting.data.fe_catch22_pca import FECatch22PCA
from kaggle_m5_forecasting.data.fe_weather import FEWeather
from kaggle_m5_forecasting.data.fe_stock import FEStock
from kaggle_m5_forecasting.data.fe_unemployment import FEUnemployment
from kaggle_m5_forecasting.data.target_encoding import TEValData, TEData
from kaggle_m5_forecasting.data.split_data import SplitValData, SplitData
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.config import Config

from typing import List, Tuple
from tqdm.autonotebook import tqdm
import pandas as pd


class CombineValFeatures(M5):
    def requires(self):
        return dict(
            sp_idxs=SplitValData(),
            data=MakeData(),
            fe_price_rolling=FEPriceRolling(),
            fe_price_change=FEPriceChange(),
            fe_price_basic=FEPriceBasic(),
            fe_shift=FEShift(),
            fe_rolling_mean=FERollingMean(),
            fe_rolling_group_mean=FERollingGroupMean(),
            fe_rolling_group_std=FERollingGroupStd(),
            fe_rolling_std=FERollingStd(),
            fe_rolling_skew=FERollingSkew(),
            fe_rolling_kurt=FERollingKurt(),
            fe_catch22_pca=FECatch22PCA(),
            fe_weather=FEWeather(),
            fe_unemployment=FEUnemployment(),
            fe_stock=FEStock(),
            te_val_data=TEValData(),
        )

    def run(self):
        with timer("combine val features"):
            with timer("concat features"):
                data: pd.DataFrame = pd.concat(
                    [
                        self.load("data"),
                        self.load("fe_price_rolling"),
                        self.load("fe_price_change"),
                        self.load("fe_price_basic"),
                        self.load("fe_shift"),
                        self.load("fe_rolling_mean"),
                        self.load("fe_rolling_group_mean"),
                        self.load("fe_rolling_group_std"),
                        self.load("fe_rolling_std"),
                        self.load("fe_rolling_skew"),
                        self.load("fe_rolling_kurt"),
                        self.load("fe_weather"),
                        self.load("fe_unemployment"),
                        self.load("fe_stock"),
                        self.load("fe_catch22_pca"),
                    ],
                    axis=1,
                )
            with timer("merge target features"):
                config = Config()
                te_val_data: List[pd.DataFrame] = self.load("te_val_data")
                splits: List[Split] = []
                sp_idxs: List[SplitIndex] = self.load("sp_idxs")
                for i in tqdm(range(len(sp_idxs))):
                    sp: Split = Split()
                    data = pd.concat([data, te_val_data[i]], axis=1)
                    sp.train = data.iloc[sp_idxs[i].train, :]
                    sp.test = data.iloc[sp_idxs[i].test, :]
                    if config.CV_SAMPLE_RATE != 1:
                        sp.train = sp.train.sample(
                            int(len(sp.train) * config.CV_SAMPLE_RATE)
                        )
                    splits.append(sp)
                    print(sp.train.info())
                    data = data.drop(list(data.filter(like="fe_te_").columns), axis=1)
        self.dump(splits)


class CombineFeatures(M5):
    def requires(self):
        return dict(
            sp_idx=SplitData(),
            data=MakeData(),
            fe_price_rolling=FEPriceRolling(),
            fe_price_change=FEPriceChange(),
            fe_price_basic=FEPriceBasic(),
            fe_shift=FEShift(),
            fe_rolling_mean=FERollingMean(),
            fe_rolling_group_mean=FERollingGroupMean(),
            fe_rolling_group_std=FERollingGroupStd(),
            fe_rolling_std=FERollingStd(),
            fe_rolling_skew=FERollingSkew(),
            fe_rolling_kurt=FERollingKurt(),
            fe_catch22_pca=FECatch22PCA(),
            fe_weather=FEWeather(),
            fe_unemployment=FEUnemployment(),
            fe_stock=FEStock(),
            te_data=TEData(),
        )

    def run(self):
        with timer("combine features"):
            with timer("concat features"):
                data: pd.DataFrame = pd.concat(
                    [
                        self.load("data"),
                        self.load("fe_price_rolling"),
                        self.load("fe_price_change"),
                        self.load("fe_price_basic"),
                        self.load("fe_shift"),
                        self.load("fe_rolling_mean"),
                        self.load("fe_rolling_group_mean"),
                        self.load("fe_rolling_group_std"),
                        self.load("fe_rolling_std"),
                        self.load("fe_rolling_skew"),
                        self.load("fe_rolling_kurt"),
                        self.load("te_data"),
                        self.load("fe_catch22_pca"),
                        self.load("fe_weather"),
                        self.load("fe_unemployment"),
                        self.load("fe_stock"),
                    ],
                    axis=1,
                )
            sp_idx: SplitIndex = self.load("sp_idx")
            sp: Split = Split()
            sp.train = data.iloc[sp_idx.train, :]
            sp.test = data.iloc[sp_idx.test, :]
            print(sp.train.info())
        self.dump(sp)
