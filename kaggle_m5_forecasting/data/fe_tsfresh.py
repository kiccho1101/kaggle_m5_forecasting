import pandas as pd
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.load_data import RawData, LoadRawData
from tsfresh import extract_features


class FETSFresh(M5):
    def requires(self):
        return dict(raw=LoadRawData())

    def run(self):
        raw: RawData = self.load("raw")

        df = pd.melt(
            raw.sales_train_validation,
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            var_name="d",
            value_name="sales",
        )

        tsfresh_df = extract_features(
            df[["id", "d", "sales"]], column_id="id", column_sort="d"
        )

        self.dump(tsfresh_df)
