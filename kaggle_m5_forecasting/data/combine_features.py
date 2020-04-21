from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.base import Split, SplitIndex
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.data.fe_price_basic import FEPriceBasic
from kaggle_m5_forecasting.data.fe_price_change import FEPriceChange
from kaggle_m5_forecasting.data.fe_price_rolling import FEPriceRolling
from kaggle_m5_forecasting.data.fe_shift import FEShift
from kaggle_m5_forecasting.data.fe_rolling import (
    FERollingSum,
    FERollingMean,
    FERollingStd,
    FERollingKurt,
    FERollingSkew,
)
from kaggle_m5_forecasting.data.fe_revenue import FERevenue
from kaggle_m5_forecasting.data.target_encoding import TEValData, TEData
from kaggle_m5_forecasting.data.split_data import SplitValData, SplitData

import pandas as pd


class CombineValFeatures(M5):
    def requires(self):
        return dict(
            sp_idx=SplitValData(),
            data=MakeData(),
            fe_price_rolling=FEPriceRolling(),
            fe_price_change=FEPriceChange(),
            fe_price_basic=FEPriceBasic(),
            fe_shift=FEShift(),
            fe_rolling_sum=FERollingSum(),
            fe_rolling_mean=FERollingMean(),
            fe_rolling_std=FERollingStd(),
            fe_rolling_skew=FERollingSkew(),
            fe_rolling_kurt=FERollingKurt(),
            fe_revenue=FERevenue(),
            te_val_data=TEValData(),
        )

    def run(self):
        data: pd.DataFrame = pd.concat(
            [
                self.load("data"),
                self.load("fe_price_rolling"),
                self.load("fe_price_change"),
                self.load("fe_price_basic"),
                self.load("fe_shift"),
                self.load("fe_rolling_sum"),
                self.load("fe_rolling_mean"),
                self.load("fe_rolling_std"),
                self.load("fe_rolling_skew"),
                self.load("fe_rolling_kurt"),
                self.load("fe_revenue"),
                self.load("te_val_data"),
            ],
            axis=1,
        )
        sp_idx: SplitIndex = self.load("sp_idx")
        sp: Split = Split()
        sp.train = data.iloc[sp_idx.train, :]
        sp.test = data.iloc[sp_idx.test, :]
        self.dump(sp)


class CombineFeatures(M5):
    def requires(self):
        return dict(
            sp_idx=SplitData(),
            data=MakeData(),
            fe_price_rolling=FEPriceRolling(),
            fe_price_change=FEPriceChange(),
            fe_price_basic=FEPriceBasic(),
            fe_shift=FEShift(),
            fe_rolling_sum=FERollingSum(),
            fe_rolling_mean=FERollingMean(),
            fe_rolling_std=FERollingStd(),
            fe_rolling_skew=FERollingSkew(),
            fe_rolling_kurt=FERollingKurt(),
            fe_revenue=FERevenue(),
            te_val_data=TEValData(),
        )

    def run(self):
        data: pd.DataFrame = pd.concat(
            [
                self.load("data"),
                self.load("fe_price_rolling"),
                self.load("fe_price_change"),
                self.load("fe_price_basic"),
                self.load("fe_shift"),
                self.load("fe_rolling_sum"),
                self.load("fe_rolling_mean"),
                self.load("fe_rolling_std"),
                self.load("fe_rolling_skew"),
                self.load("fe_rolling_kurt"),
                self.load("fe_revenue"),
                self.load("te_val_data"),
            ],
            axis=1,
        )
        sp_idx: SplitIndex = self.load("sp_idx")
        sp: Split = Split()
        sp.train = data.iloc[sp_idx.train, :]
        sp.test = data.iloc[sp_idx.test, :]
        self.dump(sp)
