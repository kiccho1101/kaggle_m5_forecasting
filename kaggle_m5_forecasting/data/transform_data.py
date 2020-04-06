from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.fe_basic import FEBasic
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage
from tqdm import tqdm
from typing import List
import sklearn.preprocessing
import pandas as pd
import gc


class TransformData(M5):
    def requires(self):
        return FEBasic()

    def run(self):
        data: pd.DataFrame = self.load()

        with timer("label encoding"):
            cat_features: List[str] = [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
            ]
            for feature in tqdm(cat_features):
                encoder = sklearn.preprocessing.LabelEncoder()
                data[feature] = encoder.fit_transform(data[feature])
        self.dump(data)
