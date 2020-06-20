import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import pickle
import glob
from kaggle_m5_forecasting.config import Config
from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.wrmsse import WRMSSEEvaluator
from kaggle_m5_forecasting.cv_dashboard import create_dashboard
import os
import sklearn.preprocessing
from typing import Dict


@dataclass
class CVResult:
    cv_num: int
    test_pred: pd.DataFrame
    config: Optional[Config] = None
    evaluator: Optional[WRMSSEEvaluator] = None

    @staticmethod
    def _create_train_df(
        raw: RawData, train_start_d: int, train_end_d: int
    ) -> pd.DataFrame:
        train_max_d = int(
            raw.sales_train_validation.filter(like="d_")
            .iloc[:, -1]
            .name.replace("d_", "")
        )

        train_drop_cols = [f"d_{d}" for d in range(train_end_d + 1, train_max_d + 1)]

        train_df: pd.DataFrame = raw.sales_train_validation.drop(
            train_drop_cols, axis=1
        )
        assert len(train_df) == 30490
        assert "id" in train_df.columns
        assert "item_id" in train_df.columns
        return train_df

    @staticmethod
    def _create_valid_df(
        raw: RawData, test_start_d: int, test_end_d: int
    ) -> pd.DataFrame:
        valid_df: pd.DataFrame = raw.sales_train_validation[
            [f"d_{d}" for d in range(test_start_d, test_end_d + 1)]
        ]
        valid_df.reset_index(drop=True, inplace=True)
        assert valid_df.shape == (30490, 28)
        return valid_df

    @staticmethod
    def _create_calendar_df(
        raw: RawData, train_start_d: int, train_end_d: int
    ) -> pd.DataFrame:
        calendar_df: pd.DataFrame = raw.calendar.iloc[
            train_start_d - 1 : train_end_d, :
        ]
        return calendar_df

    @staticmethod
    def _create_prices_df(
        raw: RawData, train_start_d: int, train_end_d: int
    ) -> pd.DataFrame:
        prices_start = raw.calendar["wm_yr_wk"][train_start_d - 1]
        prices_end = raw.calendar["wm_yr_wk"][train_end_d - 1]
        prices_df: pd.DataFrame = raw.sell_prices.loc[
            (raw.sell_prices.wm_yr_wk >= prices_start)
            & (raw.sell_prices.wm_yr_wk <= prices_end),
            :,
        ]
        return prices_df

    def _get_valid_pred_df(self, raw: RawData) -> pd.DataFrame:
        cat_encoders: Dict[str, sklearn.preprocessing.LabelEncoder] = pickle.load(
            open(
                os.path.join(os.path.dirname(__file__), "..", "cat_encoders.pkl"), "rb"
            )
        )
        valid_pred_df: pd.DataFrame = self.test_pred[
            ["item_id", "store_id", "d", "sales"]
        ]
        valid_pred_df["item_id"] = valid_pred_df["item_id"].apply(
            lambda x: cat_encoders["item_id"].classes_[x]
        )
        valid_pred_df["store_id"] = self.test_pred["store_id"].apply(
            lambda x: cat_encoders["store_id"].classes_[x]
        )
        valid_pred_df = valid_pred_df.set_index(["item_id", "store_id", "d"]).unstack()
        valid_pred_df.columns = valid_pred_df.columns.droplevel()
        valid_pred_df = valid_pred_df.loc[
            zip(
                raw.sales_train_validation.item_id, raw.sales_train_validation.store_id
            ),
            :,
        ]
        return valid_pred_df

    def get_evaluator(self, raw: RawData) -> WRMSSEEvaluator:
        train_start_d = self.config.START_DAY
        train_end_d = int(self.test_pred["d"].min()) - 1
        test_start_d = self.test_pred["d"].min()
        test_end_d = self.test_pred["d"].max()

        self.train_df = self._create_train_df(raw, train_start_d, train_end_d)
        self.valid_df = self._create_valid_df(raw, test_start_d, test_end_d)

        # self.calendar_df = self._create_calendar_df(raw, train_start_d, train_end_d)
        # self.prices_df = self._create_prices_df(raw, train_start_d, train_end_d)

        self.calendar_df = raw.calendar
        self.prices_df = raw.sell_prices

        self.evaluator = WRMSSEEvaluator(
            self.train_df, self.valid_df, self.calendar_df, self.prices_df
        )

        self.valid_pred_df = self._get_valid_pred_df(raw)
        self.evaluator.score(self.valid_pred_df.values)

        return self.evaluator

    # def create_dashboard(self, raw: RawData, save_path: str):
    #     if self.evaluator is not None:
    #         create_dashboard(self.evaluator, raw, save_path)


class CVResults:
    timestamp: str = ""
    results: List[CVResult] = []

    def from_timestamp(self, timestamp: str):
        self.timestamp = timestamp
        self.results = []
        for cv_dir in glob.glob(f"./../../output/cv/{self.timestamp}/*"):
            cv_num = int(cv_dir.split("/")[-1])
            test_pred = pickle.load(open(f"{cv_dir}/test_pred.pkl", "rb"))
            config = pickle.load(open(f"{cv_dir}/config.pkl", "rb"))
            self.results.append(
                CVResult(cv_num=cv_num, test_pred=test_pred, config=config)
            )
        return self
