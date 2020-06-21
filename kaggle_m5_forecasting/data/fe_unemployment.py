import pandas as pd
from typing import Dict
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.load_data import LoadRawData, RawData
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.utils import timer


def read_unemployment_data(
    date_range: pd.DataFrame, external_data_path: str = "./external_data"
) -> pd.DataFrame:
    files: Dict[str, int] = {
        "CA.csv": 0,
        "TX.csv": 1,
        "WI.csv": 2,
    }

    with timer("load unemployment data"):
        unemployment: pd.DataFrame = pd.DataFrame()
        for file_name, state_id in files.items():
            _tmp_unemployment = pd.read_csv(
                f"{external_data_path}/unemployment/{file_name}"
            )
            _tmp_unemployment["date"] = pd.to_datetime(
                _tmp_unemployment["DATE"]
            ).dt.strftime("%Y-%m-%d")
            _tmp_unemployment.drop("DATE", axis=1, inplace=True)
            _tmp_unemployment.rename(
                {"{}UR".format(file_name.replace(".csv", "")): "fe_unemployment"},
                axis=1,
                inplace=True,
            )
            _tmp_unemployment = date_range.merge(
                _tmp_unemployment, on="date", how="left"
            )
            _tmp_unemployment["fe_unemployment"] = _tmp_unemployment[
                "fe_unemployment"
            ].interpolate()
            _tmp_unemployment["fe_unemployment"] = _tmp_unemployment[
                "fe_unemployment"
            ].fillna(method="bfill")
            _tmp_unemployment["state_id"] = state_id
            unemployment = pd.concat([unemployment, _tmp_unemployment], axis=0)
            del _tmp_unemployment
    return unemployment


class FEUnemployment(M5):
    def requires(self):
        return dict(raw=LoadRawData(), data=MakeData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")

        raw.calendar["d"] = raw.calendar["d"].map(lambda d: int(d.replace("d_", "")))

        unemployment = read_unemployment_data(date_range=raw.calendar[["date"]])

        with timer("merge data"):
            data = data.merge(raw.calendar[["d", "date"]], on="d", how="left").merge(
                unemployment, on=["date", "state_id"], how="left"
            )

        df = data.filter(like="fe_unemployment")
        print(df.info())

        self.dump(df)
