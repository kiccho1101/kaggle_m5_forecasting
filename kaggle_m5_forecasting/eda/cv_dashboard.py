# %%
import numpy as np

import sys
import os

sys.path.append(os.getcwd() + "/../..")
from kaggle_m5_forecasting.cv_result import CVResults
from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.cv_dashboard import create_dashboard
from thunderbolt import Thunderbolt

TIMESTAMP = "2020-06-16_08:27:40"

cv = CVResults().from_timestamp(TIMESTAMP)
tb = Thunderbolt("./../../resource")
raw: RawData = tb.get_data("LoadRawData")

# %%
CV_NUM = 0

evaluator = cv.results[CV_NUM].get_evaluator(raw)
create_dashboard(evaluator, raw)
# %%
len(evaluator.all_scores)
