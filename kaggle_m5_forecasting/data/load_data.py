from logging import getLogger


from typing import Dict
from tqdm import tqdm
import swifter
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage
import pandas as pd
import numpy as np

logger = getLogger(__name__)


class RawData:
    def __init__(self):
        self.calendar: pd.DataFrame = pd.DataFrame()
        self.sales_train_validation: pd.DataFrame = pd.DataFrame()
        self.sample_submission: pd.DataFrame = pd.DataFrame()
        self.sell_prices: pd.DataFrame = pd.DataFrame()


class LoadRawData(M5):
    def run(self):
        d = RawData()

        with timer("load calendar.csv"):
            d.calendar = pd.read_csv("./m5-forecasting-accuracy/calendar.csv").pipe(
                reduce_mem_usage
            )

        with timer("load sales_train_validation.csv"):
            d.sales_train_validation = pd.read_csv(
                "./m5-forecasting-accuracy/sales_train_validation.csv"
            ).pipe(reduce_mem_usage)

        with timer("load sample_submission.csv"):
            d.sample_submission = pd.read_csv(
                "./m5-forecasting-accuracy/sample_submission.csv"
            ).pipe(reduce_mem_usage)

        with timer("load sell_prices.csv"):
            d.sell_prices = pd.read_csv(
                "./m5-forecasting-accuracy/sell_prices.csv"
            ).pipe(reduce_mem_usage)

        self.dump(d)
