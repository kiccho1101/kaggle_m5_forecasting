import pandas as pd
from typing import Dict
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.load_data import LoadRawData, RawData
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.utils import timer


def read_weather_data(external_data_path: str = "./external_data") -> pd.DataFrame:
    files: Dict[str, int] = {
        "californiaw.csv": 0,
        "texasw.csv": 1,
        "wisconsinw.csv": 2,
    }

    with timer("load weather data"):
        weather = pd.DataFrame()
        for file_name, state_id in files.items():
            _tmp_weather = pd.read_csv(f"{external_data_path}/weather/{file_name}")
            _tmp_weather["state_id"] = state_id
            _tmp_weather["date_time"] = pd.to_datetime(
                _tmp_weather["date_time"]
            ).dt.strftime("%Y-%m-%d")
            weather = pd.concat([weather, _tmp_weather], axis=0)
            del _tmp_weather
        weather.columns = [
            f"fe_weather_{col}" if col not in ["date_time", "state_id"] else col
            for col in weather.columns
        ]
    return weather


class FEWeather(M5):
    def requires(self):
        return dict(raw=LoadRawData(), data=MakeData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")

        raw.calendar["d"] = raw.calendar["d"].map(lambda d: int(d.replace("d_", "")))
        raw.calendar["date_time"] = raw.calendar["date"]

        weather = read_weather_data()
        weather = weather[
            [
                "fe_weather_mintempC",
                "fe_weather_maxtempC",
                "fe_weather_humidity",
                "fe_weather_sunHour",
                "fe_weather_cloudcover",
            ]
        ]

        with timer("merge data"):
            data = data.merge(
                raw.calendar[["d", "date_time"]], on="d", how="left"
            ).merge(weather, on=["date_time", "state_id"], how="left")

        df = data.filter(like="fe_weather_")
        print(df.info())

        self.dump(df)
