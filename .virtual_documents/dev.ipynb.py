import pandas as pd
import numpy as np
from kaggle_m5_forecasting import RawData
from thunderbolt import Thunderbolt
tb = Thunderbolt("./resource")
# data: pd.DataFrame = tb.get_data("MakeData")
raw: RawData = tb.get_data("LoadRawData")


raw.calendar[["event_name_1", "event_type_1"]].drop_duplicates()
