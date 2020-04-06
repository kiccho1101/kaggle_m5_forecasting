import lightgbm as lgb
import sklearn.metrics
import pandas as pd
import numpy as np
import datetime
import glob
import gc
from typing import List
from kaggle_m5_forecasting import M5, TransformData, LoadRawData, RawData
from kaggle_m5_forecasting.utils import timer


features: List[str] = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "sell_price",
    "price_max",
    "price_min",
    "price_std",
    "price_mean",
    "price_norm",
    "price_nunique",
    "item_nunique",
    "price_momentum",
    "price_momentum_m",
    "price_momentum_y",
    "event_name_1",
    "event_type_1",
    # "event_name_2",
    # "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "tm_d",
    "tm_w",
    "tm_m",
    "tm_y",
    "tm_wm",
    "tm_dw",
    "tm_w_end",
    "rolling_skew_t30",
    "rolling_kurt_t30",
    "price_change_t1",
    "price_change_t365",
    "rolling_price_std_t7",
    "rolling_price_std_t30",
] + [
    f"rolling_{method}_t{days}"
    for method in ["mean", "std", "max"]
    for days in [7, 30, 60, 90, 180]
] + [
    f"shift_t{days}" for days in list(range(28, 35)) + list(range(338, 340))
]

params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "regression",
    "n_jobs": -1,
    "seed": 42,
    "learning_rate": 0.1,
    "lambda": 0.1,
    "bagging_fraction": 0.66,
    "bagging_freq": 2,
    "colsample_bytree": 0.75,
}


class LGBMVal(M5):
    def requires(self):
        return dict(data=TransformData(), raw=LoadRawData())

    def run(self):

        data: pd.DataFrame = self.load("data")

        with timer("split into train, val, test"):
            train = data[data.d <= 1885]
            val = data[(data.d > 1885) & (data.d <= 1913)]
            del data
            gc.collect()

        train_set = lgb.Dataset(train[features], train["sales"])
        val_set = lgb.Dataset(val[features], val["sales"])

        with timer("train lgbm model"):
            model = lgb.train(
                params,
                train_set,
                num_boost_round=2000,
                early_stopping_rounds=200,
                valid_sets=[train_set, val_set],
                verbose_eval=100,
            )
        with timer("predict val"):
            val_pred = model.predict(val[features], num_iteration=model.best_iteration)
            val_score = np.sqrt(
                sklearn.metrics.mean_squared_error(val_pred, val["sales"])
            )
            print("score:", val_score)

        self.dump(model)


class LGBMSubmission(M5):
    def requires(self):
        return dict(data=TransformData(), raw=LoadRawData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")
        with timer("split into train, val, test"):
            train = data[data.d <= 1913]
            test = data[(data.d > 1913)]

        train_set = lgb.Dataset(train[features], train["sales"])
        del data
        gc.collect()

        with timer("train lgbm model"):
            model = lgb.train(
                params,
                train_set,
                num_boost_round=2000,
                early_stopping_rounds=200,
                valid_sets=[train_set],
                verbose_eval=100,
            )

        with timer("predict test"):
            test["sales"] = model.predict(test[features])

            submission_file_prefix: str = "./output/submission/submission_{}_lgbm_regression".format(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )
            submission_no = len(glob.glob(submission_file_prefix + "_*.csv")) + 1
            submission_file_name = "{}_{}.csv".format(
                submission_file_prefix, submission_no
            )

        with timer("create submission file: {}".format(submission_file_name)):
            predictions = test[["id", "d", "sales"]]
            predictions = pd.pivot(
                predictions, index="id", columns="d", values="sales"
            ).reset_index()
            predictions.columns = ["id"] + [f"F{i}" for i in range(1, 29)]
            evaluation_rows = [
                row for row in raw.sample_submission["id"] if "evaluation" in row
            ]
            evaluation = raw.sample_submission[
                raw.sample_submission["id"].isin(evaluation_rows)
            ]

            validation = raw.sample_submission[["id"]].merge(predictions, on="id")
            final = pd.concat([validation, evaluation])

        for i in range(1, 29):
            final["F" + str(i)] *= 1.04
        final.to_csv(submission_file_name, index=False)
