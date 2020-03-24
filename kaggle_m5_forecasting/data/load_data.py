from logging import getLogger


from kaggle_m5_forecasting import M5
import pandas as pd

logger = getLogger(__name__)


class Data:
    def __init__(self):
        self.calendar: pd.DataFrame = pd.DataFrame()
        self.sales_train_validation: pd.DataFrame = pd.DataFrame()
        self.sample_submission: pd.DataFrame = pd.DataFrame()
        self.sell_prices: pd.DataFrame = pd.DataFrame()


class LoadData(M5):
    def run(self):
        d = Data()

        logger.info("loading calendar.csv")
        d.calendar = pd.read_csv("./m5-forecasting-accuracy/calendar.csv")
        logger.info("loaded calendar.csv")

        logger.info("loading sales_train_validation.csv")
        d.sales_train_validation = pd.read_csv(
            "./m5-forecasting-accuracy/sales_train_validation.csv"
        )
        logger.info("loaded sales_train_validation.csv")

        logger.info("loading sample_submission.csv")
        d.sample_submission = pd.read_csv(
            "./m5-forecasting-accuracy/sample_submission.csv"
        )
        logger.info("loaded sample_submission.csv")

        logger.info("loading sell_prices.csv")
        d.sell_prices = pd.read_csv("./m5-forecasting-accuracy/sell_prices.csv")
        logger.info("loaded sell_prices.csv")

        self.dump(d)
