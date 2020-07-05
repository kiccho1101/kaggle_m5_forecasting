import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from kaggle_m5_forecasting.events import events
from kaggle_m5_forecasting import M5

from kaggle_m5_forecasting.data.load_data import RawData, LoadRawData
from kaggle_m5_forecasting.data.make_data import MakeData


class FEEventStrength(M5):
    def requires(self):
        return dict(raw=LoadRawData(), data=MakeData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")

        raw.calendar["d"] = raw.calendar["d"].str.replace("d_", "").astype(int)
        group = ["store_id", "dept_id"]
        df = data.groupby(["d"] + group)["sales"].mean().reset_index()
        df = df.merge(raw.calendar[["d", "wday"]], on="d", how="left")
        df["fe_rolling_{}_dw_mean_1_4".format("_".join(group))] = df.groupby(
            group + ["wday"]
        )["sales"].transform(lambda x: x.shift(1).rolling(4).mean())
        df["{}_sales_mean".format("_".join(group))] = df["sales"]
        df.drop("sales", axis=1, inplace=True)

        df["store_id_dept_id_diff"] = (
            (
                df["store_id_dept_id_sales_mean"]
                - df["fe_rolling_store_id_dept_id_dw_mean_1_4"]
            )
            / df["fe_rolling_store_id_dept_id_dw_mean_1_4"]
            * 100
        )
        df = df.merge(raw.calendar[["d", "date"]], on="d", how="left")
        df["tm_w"] = pd.to_datetime(df["date"]).dt.week
        data = data.merge(raw.calendar[["d", "date"]], on="d", how="left")

        event_strength = pd.DataFrame()

        event_dates_all = []
        for event_name, event_dates in tqdm(events.items()):
            event_dates_all += event_dates
            _tmp = (
                df[df["date"].isin(event_dates)]
                .groupby(group)["store_id_dept_id_diff"]
                .apply(lambda x: np.nanmean(x))
                .reset_index()
            )
            for event_date in event_dates:
                _tmp["date"] = event_date
                event_strength = pd.concat([event_strength, _tmp], axis=0)

        fillna = df[group + ["tm_w"]].merge(
            df[~df["date"].isin(event_dates_all)]
            .groupby(group + ["tm_w"])["store_id_dept_id_diff"]
            .apply(lambda x: np.nanmean(x))
            .reset_index(),
            on=group + ["tm_w"],
            how="left",
        )["store_id_dept_id_diff"]
        df = df.drop("store_id_dept_id_diff", axis=1).merge(
            event_strength, on=group + ["date"], how="left"
        )
        df["store_id_dept_id_diff"] = df["store_id_dept_id_diff"].fillna(fillna)

        df["fe_event_strength"] = df["store_id_dept_id_diff"]
        data = data[group + ["date"]].merge(df[group + ["date", "fe_event_strength"]])

        df = data[["fe_event_strength"]]

        print(df.info())

        self.dump(df)
