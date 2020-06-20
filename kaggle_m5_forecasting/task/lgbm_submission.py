import lightgbm as lgb
from lightgbm import LGBMClassifier
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
import multiprocessing
from joblib import Parallel, delayed
from kaggle_m5_forecasting import (
    M5,
    LoadRawData,
    RawData,
    config,
)
from kaggle_m5_forecasting.base import Split
from kaggle_m5_forecasting.data.combine_features import (
    CombineValFeatures,
    CombineFeatures,
)
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.fe_rolling import make_rolling_for_test


class LGBMSubmission(M5):
    def requires(self):
        return dict(sp=CombineFeatures(), raw=LoadRawData())

    def run(self):
        raw: RawData = self.load("raw")
        sp: Split = self.load("sp")
        sp.train = sp.train[(sp.train.fe_rolling_sum_t0_30 > config.MIN_SUM)][
            config.features + [config.TARGET]
        ]
        print("train shape:", sp.train.shape)
        print(
            "number of NaN:",
            sp.train.isna().sum(axis=0).where(lambda x: x > 0).dropna().sort_values(),
        )
        if config.DROP_NA:
            sp.train.dropna(inplace=True)
            print("train shape:", sp.train.shape)
        train_set = lgb.Dataset(sp.train[config.features], sp.train[config.TARGET])
        sp.train = pd.DataFrame()
        gc.collect()

        try:
            mlflow.end_run()
        except Exception:
            pass

        mlflow.lightgbm.autolog()
        with mlflow.start_run(run_name=""):
            with timer("train lgbm model", mlflow_on=True):
                mlflow.log_param("MIN_SUM", config.MIN_SUM)
                mlflow.log_param("MAX_LAGS", config.MAX_LAGS)
                mlflow.log_param("start_day", config.START_DAY)
                mlflow.log_param("SEED", config.SEED)
                mlflow.log_param("features", str(config.features))
                mlflow.log_param("DROP_NA", config.DROP_NA)

                # model = lgb.Booster(
                #     model_file="./mlruns/0/89f225d1fc624c69af3443e2982c8ddb/artifacts/model/model.lgb"
                # )
                if config.TARGET == "sales":
                    model = lgb.train(
                        config.lgbm_params,
                        train_set,
                        num_boost_round=config.num_boost_round,
                        verbose_eval=50,
                        categorical_feature=config.lgbm_cat_features,
                        valid_sets=[train_set],
                    )
                elif config.TARGET == "sales_is_zero":
                    params = config.lgbm_params
                    if config.TARGET == "sales_is_zero":
                        params["objective"] = "binary"
                        params["metric"] = "binary_error,auc"
                        params["random_state"] = params["seed"]
                        params["min_child_samples"] = params["min_data_in_leaf"]
                        params["n_estimators"] = config.num_boost_round
                        model = LGBMClassifier(**params)
                        model.fit(
                            sp.train[config.features],
                            sp.train[config.TARGET],
                            verbose=50,
                            categorical_feature=config.lgbm_cat_features,
                        )

            with timer("predict test", mlflow_on=True):

                for d in tqdm(range(1914, 1914 + 28)):
                    sp.test = make_rolling_for_test(sp.test, d, config.features)
                    if config.TARGET == "sales":
                        sp.test.loc[sp.test.d == d, config.TARGET] = model.predict(
                            sp.test.loc[sp.test.d == d, config.features]
                        )

            with timer("create submission file"):
                submission_file_prefix: str = "./output/submission/submission_{}_lgbm".format(
                    datetime.datetime.now().strftime("%Y-%m-%d")
                )
                submission_no = len(glob.glob(submission_file_prefix + "_*.csv")) + 1
                submission_file_name = "{}_{}.csv".format(
                    submission_file_prefix, submission_no
                )

                mlflow.log_param("submission_file_name", submission_file_name)

                predictions = sp.test[sp.test.d > 1913][["id", "d", config.TARGET]]
                predictions = pd.pivot(
                    predictions, index="id", columns="d", values=config.TARGET
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

        final["F1"] *= 3215 / 3186
        final["F2"] *= final["F2"].sum() / (final["F2"].sum() + 119)
        final["F3"] *= 1913 / 1923
        final["F4"] /= 1.0073
        final["F5"] *= 0.995874
        final["F6"] *= 1.000376
        final["F7"] *= 0.995635
        final["F8"] *= 0.9988
        for i in range(9, 20):
            if i != 11:
                final["F" + str(i)] *= 1.01
        for i in range(20, 29):
            final["F" + str(i)] *= 1.02
        final.to_csv(submission_file_name, index=False)
        self.dump(model)
