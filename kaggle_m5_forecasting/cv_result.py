import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import pickle
import glob
from kaggle_m5_forecasting.config import Config
from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.wrmsse import WRMSSEEvaluator


@dataclass
class CVResult:
    cv_num: int
    test_pred: pd.DataFrame
    evaluator: Optional[WRMSSEEvaluator] = None
    config: Optional[Config] = None

    @staticmethod
    def _create_train_df(raw: RawData, cv_start_d: int, cv_end_d: int) -> pd.DataFrame:
        train_max_d = int(
            raw.sales_train_validation.filter(like="d_")
            .iloc[:, -1]
            .name.replace("d_", "")
        )
        train_drop_cols = [f"d_{d}" for d in range(1, cv_start_d)]
        train_drop_cols += [f"d_{d}" for d in range(cv_end_d + 1, train_max_d + 1)]
        train_df: pd.DataFrame = raw.sales_train_validation.drop(
            train_drop_cols, axis=1
        )
        assert len(train_df) == 30490
        assert "id" in train_df.columns
        assert "item_id" in train_df.columns
        return train_df

    @staticmethod
    def _create_valid_df(test_pred: pd.DataFrame) -> pd.DataFrame:
        valid_df: pd.DataFrame = pd.pivot(
            test_pred, index="id", columns="d", values="sales"
        )
        valid_df.columns = [f"d_{d}" for d in valid_df.columns]
        valid_df.reset_index(drop=True, inplace=True)
        assert valid_df.shape == (30490, 28)
        return valid_df

    @staticmethod
    def _create_calendar_df(
        raw: RawData, cv_start_d: int, cv_end_d: int
    ) -> pd.DataFrame:
        calendar_df: pd.DataFrame = raw.calendar.iloc[cv_start_d - 1 : cv_end_d, :]
        return calendar_df

    @staticmethod
    def _create_prices_df(raw: RawData, cv_start_d: int, cv_end_d: int) -> pd.DataFrame:
        prices_start = raw.calendar["wm_yr_wk"][cv_start_d - 1]
        prices_end = raw.calendar["wm_yr_wk"][cv_end_d - 1]
        prices_df: pd.DataFrame = raw.sell_prices.loc[
            (raw.sell_prices.wm_yr_wk >= prices_start)
            & (raw.sell_prices.wm_yr_wk <= prices_end),
            :,
        ]
        return prices_df

    def wrmsse(self, raw: RawData):
        cv_start_d = self.config.START_DAY
        cv_end_d = int(self.test_pred["d"].min()) - 1

        train_df = self._create_train_df(raw, cv_start_d, cv_end_d)
        valid_df = self._create_valid_df(self.test_pred)
        calendar_df = self._create_calendar_df(raw, cv_start_d, cv_end_d)
        prices_df = self._create_prices_df(raw, cv_start_d, cv_end_d)
        self.evaluator = WRMSSEEvaluator(train_df, valid_df, calendar_df, prices_df)


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
