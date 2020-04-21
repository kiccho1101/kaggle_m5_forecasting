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
from kaggle_m5_forecasting.metric import WRMSSEEvaluator, calc_metrics


class LGBMVal(M5):
    def requires(self):
        return dict(sp=CombineValFeatures(), raw=LoadRawData())

    def run(self):
        sp: Split = self.load("sp")
        raw: RawData = self.load("raw")
        sp.train = sp.train[(sp.train.fe_rolling_sum_t0_30 >= config.MIN_SUM)]
        print("train shape:", sp.train.shape)
        train_set = lgb.Dataset(sp.train[config.features], sp.train["sales"])
        val_set = lgb.Dataset(
            sp.test[sp.test.d > 1885][config.features],
            sp.test[sp.test.d > 1885]["sales"],
        )
        try:
            mlflow.end_run()
        except Exception:
            pass

        if mlflow.get_experiment_by_name("validation") is None:
            mlflow.create_experiment("validation")

        with mlflow.start_run(
            experiment_id=mlflow.get_experiment_by_name("validation").experiment_id,
            run_name="",
        ):
            mlflow.lightgbm.autolog()
            mlflow.log_param("MIN_SUM", config.MIN_SUM)
            mlflow.log_param("MAX_LAGS", config.MAX_LAGS)
            mlflow.log_param("start_day", config.START_DAY)
            mlflow.log_param("SEED", config.SEED)
            mlflow.log_param("features", str(config.features))

            with timer("lgbm train", mlflow_on=True):
                model = lgb.train(
                    config.lgbm_params,
                    train_set,
                    num_boost_round=config.num_boost_round,
                    verbose_eval=50,
                    categorical_feature=config.lgbm_cat_features,
                    early_stopping_rounds=20,
                    valid_sets=[train_set, val_set],
                )

                # model = lgb.Booster(
                #     model_file="./mlruns/1/11b33d7059494ca2b2f3ffd72df90a95/artifacts/model/model.lgb"
                # )

            with timer("lgbm predict", mlflow_on=True):
                test_true = sp.test.copy()
                test_pred = sp.test.copy()
                for d in tqdm(range(1914 - 28, 1914)):
                    print(d)
                    test_pred = make_rolling_for_test(test_pred, d, config.features)
                    print(
                        (
                            test_true.loc[(test_true.d == d), "fe_rolling_mean_t7_7"]
                            - test_pred.loc[(test_pred.d == d), "fe_rolling_mean_t7_7"]
                        ).mean()
                    )
                    print(
                        (
                            test_true.loc[(test_true.d == d), "shift_t7"]
                            - test_pred.loc[(test_pred.d == d), "shift_t7"]
                        ).mean()
                    )
                    print(
                        (
                            test_true.loc[(test_true.d == d), "fe_rolling_mean_t7_30"]
                            - test_pred.loc[(test_pred.d == d), "fe_rolling_mean_t7_30"]
                        ).mean()
                    )
                    test_pred.loc[test_pred.d == d, "sales"] = model.predict(
                        test_pred.loc[test_pred.d == d, config.features]
                    )

            with timer("calc metrics", mlflow_on=True):
                wrmsse, rmse, mae = calc_metrics(
                    raw,
                    test_pred[(test_pred.d > 1885) & (test_pred.d < 1914)],
                    test_true[(test_true.d > 1885) & (test_true.d < 1914)],
                )

            print("=================================")
            print("WRMSSE", wrmsse)
            print("RMSE", rmse)
            print("MAE", mae)
            print("=================================")
            mlflow.log_metric("WRMSSE", wrmsse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            self.dump(
                {
                    "test_pred": test_pred,
                    "test_true": test_true,
                    "model": model,
                    "features": config.features,
                }
            )


class LGBMSubmission(M5):
    def requires(self):
        return dict(sp=CombineFeatures(), raw=LoadRawData())

    def run(self):
        raw: RawData = self.load("raw")
        sp: Split = self.load("sp")
        sp.train = sp.train[(sp.train.fe_rolling_sum_t0_30 > config.MIN_SUM)]
        train_set = lgb.Dataset(sp.train[config.features], sp.train["sales"])
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

                # model = lgb.Booster(
                #     model_file="./mlruns/0/ada288a146964aae9529808ee1a489a8/artifacts/model/model.lgb"
                # )

                model = lgb.train(
                    config.lgbm_params,
                    train_set,
                    num_boost_round=config.num_boost_round,
                    verbose_eval=50,
                    categorical_feature=config.lgbm_cat_features,
                    valid_sets=[train_set],
                )

            with timer("predict test", mlflow_on=True):

                for d in tqdm(range(1914, 1914 + 28)):
                    sp.test = make_rolling_for_test(sp.test, d, config.features)
                    sp.test.loc[sp.test.d == d, "sales"] = model.predict(
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

                predictions = sp.test[sp.test.d > 1913][["id", "d", "sales"]]
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
            final["F" + str(i)] *= 1.025

        final.to_csv(submission_file_name, index=False)
        self.dump(model)
