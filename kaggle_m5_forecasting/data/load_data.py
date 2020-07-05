from logging import getLogger


from typing import Dict
from tqdm import tqdm
import swifter
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage
from kaggle_m5_forecasting import events
import pandas as pd
import numpy as np
from dataclasses import dataclass


logger = getLogger(__name__)


@dataclass
class RawData:
    calendar: pd.DataFrame = pd.DataFrame()
    sales_train_validation: pd.DataFrame = pd.DataFrame()
    sample_submission: pd.DataFrame = pd.DataFrame()
    sell_prices: pd.DataFrame = pd.DataFrame()


class LoadRawData(M5):
    def run(self):
        d = RawData()

        with timer("load calendar.csv"):
            d.calendar = pd.read_csv("./m5-forecasting-accuracy/calendar.csv").pipe(
                reduce_mem_usage
            )

        with timer("load sales_train_validation.csv"):
            d.sales_train_validation = pd.read_csv(
                "./m5-forecasting-accuracy/sales_train_evaluation.csv"
            ).pipe(reduce_mem_usage)

        # with timer("convert christmas data to rmean"):
        #     for d_str in d.calendar[d.calendar["date"].isin(events.christmas_dates)][
        #         "d"
        #     ]:
        #         d_int = int(d_str.replace("d_", ""))
        #         d.sales_train_validation[d_str] = d.sales_train_validation[
        #             [f"d_{i}" for i in range(d_int - 15, d_int + 15) if i != d_int]
        #         ].apply(lambda row: row.mean(), axis=1)

        with timer("load sample_submission.csv"):
            d.sample_submission = pd.read_csv(
                "./m5-forecasting-accuracy/sample_submission.csv"
            ).pipe(reduce_mem_usage)

        with timer("load sell_prices.csv"):
            d.sell_prices = pd.read_csv(
                "./m5-forecasting-accuracy/sell_prices.csv"
            ).pipe(reduce_mem_usage)

        self.dump(d)
