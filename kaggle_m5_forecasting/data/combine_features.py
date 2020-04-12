from kaggle_m5_forecasting.utils import timer, reduce_mem_usage
from tqdm import tqdm
from typing import List
from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.data.fe_price_basic import FEPriceBasic
from kaggle_m5_forecasting.data.fe_shift import FEShift
from kaggle_m5_forecasting.data.fe_rolling import (
    FERollingSum,
    FERollingMean,
    FERollingMeanDW,
    FERollingStd,
    FERollingSkew,
    FERollingKurt,
)
from kaggle_m5_forecasting.data.fe_price_change import FEPriceChange
from kaggle_m5_forecasting.data.fe_price_rolling import FEPriceRolling
from kaggle_m5_forecasting.data.fe_revenue import FERevenue
from kaggle_m5_forecasting.data.fe_target import FETarget
import sklearn.preprocessing
import pandas as pd
import gc


class CombineFeatures(M5):
    def requires(self):
        return dict(
            data=MakeData(),
            fe_price_basic=FEPriceBasic(),
            fe_shift=FEShift(),
            fe_rolling_sum=FERollingSum(),
            fe_rolling_mean=FERollingMean(),
            fe_rolling_mean_dw=FERollingMeanDW(),
            fe_rolling_std=FERollingStd(),
            fe_rolling_kurt=FERollingKurt(),
            fe_rolling_skew=FERollingSkew(),
            fe_price_change=FEPriceChange(),
            fe_price_rolling=FEPriceRolling(),
            fe_revenue=FERevenue(),
            fe_target=FETarget(),
        )

    def run(self):
        data: pd.DataFrame = pd.concat(
            [
                self.load("data"),
                self.load("fe_price_basic"),
                self.load("fe_shift"),
                self.load("fe_rolling_sum"),
                self.load("fe_rolling_mean"),
                self.load("fe_rolling_mean_dw"),
                self.load("fe_rolling_std"),
                self.load("fe_rolling_kurt"),
                self.load("fe_rolling_skew"),
                self.load("fe_price_change"),
                self.load("fe_price_rolling"),
                self.load("fe_revenue"),
                self.load("fe_target"),
            ],
            axis=1,
        )

        with timer("label encoding"):
            cat_features: List[str] = [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "event_name_1_yesterday",
                "event_type_1_yesterday",
                "event_name_1_tomorrow",
                "event_type_1_tomorrow",
            ]
            for feature in tqdm(cat_features):
                encoder = sklearn.preprocessing.LabelEncoder()
                data[feature] = encoder.fit_transform(data[feature])
        self.dump(data)
