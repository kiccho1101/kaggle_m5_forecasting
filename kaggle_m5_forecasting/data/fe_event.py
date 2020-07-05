import pandas as pd
import numpy as np

from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.load_data import LoadRawData, RawData
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.events import events


class FEEvent(M5):
    def requires(self):
        return dict(raw=LoadRawData(), data=MakeData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")

        raw.calendar["d"] = raw.calendar["d"].str.replace("d_", "").astype(int)
        raw.calendar["fe_event"] = np.nan
        for event_name, event_dates in events.items():
            raw.calendar.loc[
                raw.calendar["date"].isin(event_dates), "event"
            ] = event_name
            raw.calendar["fe_event"] = (
                raw.calendar["event"].fillna("nothing").astype(str)
            )
        raw.calendar["fe_event_dw"] = (
            raw.calendar["event"].fillna(raw.calendar["wday"].astype(str)).astype(str)
        )

        data = data.merge(
            raw.calendar[["d", "fe_event", "fe_event_dw"]], on="d", how="left"
        )
        df = data.filter(like="fe_event")
        df = df.astype("category")

        print(df.info())
        self.dump(df)
