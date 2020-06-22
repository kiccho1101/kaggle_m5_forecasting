from kaggle_m5_forecasting import M5, MakeData
from kaggle_m5_forecasting.utils import timer, reduce_mem_usage, print_mem_usage
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc


class FEShift(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        with timer("make shift features"):
            for days in tqdm(list(range(5, 9)) + list(range(28, 33))):
                data[f"shift_t{days}"] = (
                    data.groupby(["id"])["sales"]
                    .transform(lambda x: x.shift(days))
                    .astype(np.float16)
                )
        df = data.filter(like="shift_t")
        print(df.info())
        self.dump(df)
