import lightgbm as lgb
import sklearn.metrics
import pandas as pd
import numpy as np
import datetime
import glob
import gc
import mlflow
import mlflow.lightgbm
from tqdm.autonotebook import tqdm
from typing import List
import pickle
from kaggle_m5_forecasting import M5, CombineFeatures, LoadRawData, RawData
from kaggle_m5_forecasting.utils import timer


START_DAY = 300
num_boost_round = 2500
MIN_SUM = 0
MAX_LAGS = 57

features: List[str] = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "event_name_1_yesterday",
    "event_type_1_yesterday",
    "event_name_1_tomorrow",
    "event_type_1_tomorrow",
    "cal_christmas_eve",
    "cal_christmas",
    "cal_blackfriday",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "tm_d",
    "tm_w",
    "tm_wday",
    "tm_m",
    "tm_y",
    "tm_quarter",
    "tm_wm",
    "tm_wy",
    "tm_dw",
    # "tm_w_end",
    "sell_price",
    "fe_price_max",
    # "fe_price_min",
    "fe_price_std",
    "fe_price_mean",
    "fe_price_discount",
    "fe_price_discount_rate",
    # "fe_price_norm",
    # "fe_price_nunique",
    # "fe_price_item_nunique",
    # "fe_price_momentum",
    # "fe_price_momentum_m",
    # "fe_price_momentum_y",
    "shift_t7",
    "shift_t28",
    "shift_t29",
    "shift_t30",
    # "shift_t338",
    "fe_rolling_mean_t7_7",
    "fe_rolling_mean_t7_30",
    "fe_rolling_mean_t28_7",
    "fe_rolling_std_t28_7",
    "fe_rolling_mean_t28_30",
    "fe_rolling_std_t28_30",
    "fe_rolling_mean_t28_90",
    "fe_rolling_mean_t28_180",
    "fe_rolling_skew_t28_30",
    "fe_rolling_kurt_t28_30",
    # "fe_rolling_mean_dw_t28_30",
    # "fe_rolling_mean_dw_store_t28_30",
    # "fe_rolling_std_item_t28_7",
    # "fe_rolling_mean_item_t28_7",
    # "fe_rolling_mean_item_t28_30",
    # "fe_rolling_std_item_t28_30",
    "fe_price_change_t1",
    "fe_price_change_t365",
    "fe_rolling_price_std_t7",
    "fe_rolling_price_std_t30",
    # "fe_target_mean",
    # "fe_target_std",
    # "fe_target_max",
]

cat_features: List[str] = [
    f
    for f in features
    if f
    in [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "event_name_1_yesterday",
        "event_type_1_yesterday",
        "event_name_1_tomorrow",
        "event_type_1_tomorrow",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
]

params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "poisson",
    "n_jobs": -1,
    "seed": 42,
    "num_leaves": 128,
    "min_data_in_leaf": 50,
    "learning_rate": 0.075,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.75,
}


class LGBMVal(M5):
    def requires(self):
        return dict(data=CombineFeatures(), raw=LoadRawData())

    def run(self):

        data: pd.DataFrame = self.load("data")

        with timer("split into train, val, test"):
            train = data[
                (data.d >= START_DAY)
                & (data.d <= 1885)
                & (data.fe_rolling_sum_t0_30 >= MIN_SUM)
            ][features + ["sales"]]
            val = data[(data.d > 1885) & (data.d <= 1913)]
            del data
            gc.collect()

        train_set = lgb.Dataset(train[features], train["sales"])
        val_set = lgb.Dataset(val[features], val["sales"])

        with timer("train lgbm model"):
            model = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                early_stopping_rounds=200,
                valid_sets=[train_set, val_set],
                verbose_eval=50,
                categorical_feature=cat_features,
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
        return dict(data=CombineFeatures(), raw=LoadRawData())

    def run(self):
        raw: RawData = self.load("raw")
        data: pd.DataFrame = self.load("data")

        np.random.seed(777)
        with timer("split into train, val, test"):
            train = data[
                (data.d >= START_DAY)
                & (data.d <= 1913)
                & (data.fe_rolling_sum_t0_30 >= MIN_SUM)
            ][features + ["sales"]]
            val = train.sample(2_000_000).copy()
            print("train shape:", train.shape)
            print("val shape:", val.shape)

        train_set = lgb.Dataset(train[features], train["sales"])
        val_set = lgb.Dataset(val[features], val["sales"])
        del train, val
        gc.collect()

        try:
            mlflow.end_run()
        except Exception:
            pass

        mlflow.lightgbm.autolog()
        with mlflow.start_run():
            with timer("train lgbm model", mlflow_on=True):
                mlflow.log_param("MIN_SUM", MIN_SUM)
                mlflow.log_param("MAX_LAGS", MAX_LAGS)
                mlflow.log_param("features", str(features))

                model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=200,
                    valid_sets=[train_set, val_set],
                    verbose_eval=50,
                    categorical_feature=cat_features,
                )

            with timer("predict test", mlflow_on=True):
                test = data[(data.d > 1913 - MAX_LAGS)][["id", "d", "sales"]].copy()
                alphas: List[float] = [1.035, 1.03, 1.025]
                # alphas: List[float] = [1.0]
                weights: List[float] = [1 / len(alphas)] * len(alphas)

                mlflow.log_param("alphas", str(alphas))
                mlflow.log_param("weights", str(weights))

                for i, (alpha, weight) in tqdm(enumerate(zip(alphas, weights))):
                    test_per_alpha = data[(data.d > 1913 - MAX_LAGS)].copy()
                    print(
                        test_per_alpha[test_per_alpha.d > 1913]["sales"].notnull().sum()
                    )
                    for tdelta in tqdm(range(0, 28)):
                        test_per_alpha["shift_t7"] = (
                            test_per_alpha[["id", "sales"]]
                            .groupby("id")["sales"]
                            .shift(7)
                        )
                        for w_size in [7, 30]:
                            test_per_alpha[f"fe_rolling_mean_t7_{w_size}"] = (
                                test_per_alpha[["id", "shift_t7"]]
                                .groupby("id")["shift_t7"]
                                .transform(lambda x: x.rolling(w_size).mean())
                            )
                        test_per_alpha.loc[
                            test_per_alpha.d == 1914 + tdelta, "sales"
                        ] = (
                            alpha
                            * model.predict(
                                test_per_alpha.loc[
                                    test_per_alpha.d == 1914 + tdelta, features
                                ]
                            )
                        )
                    if i == 0:
                        test["sales"] = (
                            weight * test_per_alpha[test_per_alpha.d > 1913]["sales"]
                        )
                    else:
                        test["sales"] += (
                            weight * test_per_alpha[test_per_alpha.d > 1913]["sales"]
                        )

            submission_file_prefix: str = "./output/submission/submission_{}_lgbm".format(
                datetime.datetime.now().strftime("%Y-%m-%d")
            )
            submission_no = len(glob.glob(submission_file_prefix + "_*.csv")) + 1
            submission_file_name = "{}_{}.csv".format(
                submission_file_prefix, submission_no
            )
            model_file_name = submission_file_name.replace(
                "submission", "model"
            ).replace(".csv", ".pkl")
            mlflow.log_param("submission_file_name", submission_file_name)
            mlflow.log_param("model_file_name", model_file_name)

        with timer("save model: {}".format(model_file_name)):
            with open(model_file_name, "wb") as f:
                pickle.dump(model, f)

        with timer("create submission file: {}".format(submission_file_name)):
            predictions = test[test.d > 1913][["id", "d", "sales"]]
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

        # for i in range(1, 29):
        #     final["F" + str(i)] *= 1.04

        final.to_csv(submission_file_name, index=False)
        self.dump(model)
