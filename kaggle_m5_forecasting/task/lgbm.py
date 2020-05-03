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


class LGBMCrossValidation(M5):
    def requires(self):
        return dict(splits=CombineValFeatures(), raw=LoadRawData())

    def run(self):
        with timer("lgbm cv"):
            splits: List[Split] = self.load("splits")
            raw: RawData = self.load("raw")
            try:
                mlflow.end_run()
            except Exception:
                pass

            if mlflow.get_experiment_by_name("cv") is None:
                mlflow.create_experiment("cv")

            wrmsses, rmses, maes = [], [], []
            experiment_id = mlflow.get_experiment_by_name("cv").experiment_id
            with mlflow.start_run(experiment_id=experiment_id, run_name=""):
                mlflow.lightgbm.autolog()
                mlflow.log_param("MIN_SUM", config.MIN_SUM)
                mlflow.log_param("MAX_LAGS", config.MAX_LAGS)
                mlflow.log_param("start_day", config.START_DAY)
                mlflow.log_param("SEED", config.SEED)
                mlflow.log_param("features", str(config.features))

                for cv_num, sp in enumerate(splits):

                    if cv_num == 2:
                        continue

                    with timer(f"CV: {cv_num}"):

                        with timer("prepare dataset"):

                            sp.train = sp.train[
                                (sp.train.fe_rolling_sum_t0_30 >= config.MIN_SUM)
                                & (sp.train.d > config.START_DAY)
                            ][config.features + ["sales"]]
                            print("train shape:", sp.train.shape)
                            print(
                                "number of NaN:",
                                sp.train.isna()
                                .sum(axis=0)
                                .where(lambda x: x > 0)
                                .dropna()
                                .sort_values(),
                            )
                            # sp.train.dropna(inplace=True)
                            # print("train shape:", sp.train.shape)
                            mlflow.log_param(f"train shape CV_{cv_num}", sp.train.shape)
                            train_set = lgb.Dataset(
                                sp.train[config.features], sp.train["sales"]
                            )
                            val_set = lgb.Dataset(
                                sp.test[sp.test.d > config.CV_START_DAYS[cv_num]][
                                    config.features
                                ],
                                sp.test[sp.test.d > config.CV_START_DAYS[cv_num]][
                                    "sales"
                                ],
                            )

                        with timer(f"lgbm train CV_{cv_num}", mlflow_on=True):
                            model = lgb.train(
                                config.lgbm_params,
                                train_set,
                                num_boost_round=config.num_boost_round,
                                verbose_eval=10,
                                categorical_feature=config.lgbm_cat_features,
                                early_stopping_rounds=10,
                                valid_sets=[val_set],
                            )

                        with timer(f"lgbm predict CV_{cv_num}", mlflow_on=True):
                            test_true = sp.test.copy()
                            test_pred = sp.test.copy()
                            for d in tqdm(
                                range(
                                    config.CV_START_DAYS[cv_num],
                                    config.CV_START_DAYS[cv_num] + 28,
                                )
                            ):
                                test_pred = make_rolling_for_test(
                                    test_pred, d, config.features
                                )
                                test_pred.loc[
                                    test_pred.d == d, "sales"
                                ] = model.predict(
                                    test_pred.loc[test_pred.d == d, config.features]
                                )
                                test_pred.loc[test_pred.d == d, "sales_is_zero"] = (
                                    test_pred.loc[test_pred.d == d] == 0
                                ).astype(np.int8)

                        with timer(f"calc metrics CV_{cv_num}", mlflow_on=True):
                            wrmsse, rmse, mae, evaluator = calc_metrics(
                                raw,
                                test_pred[
                                    (test_pred.d >= config.CV_START_DAYS[cv_num])
                                    & (test_pred.d < config.CV_START_DAYS[cv_num] + 28)
                                ],
                                test_true[
                                    (test_true.d >= config.CV_START_DAYS[cv_num])
                                    & (test_true.d < config.CV_START_DAYS[cv_num] + 28)
                                ],
                            )

                        print(f"==========CV No: {cv_num}=================")
                        print("WRMSSE", wrmsse)
                        print("RMSE", rmse)
                        print("MAE", mae)
                        print("=================================")
                        mlflow.log_metric(f"WRMSSE_{cv_num}", wrmsse)
                        mlflow.log_metric(f"RMSE_{cv_num}", rmse)
                        mlflow.log_metric(f"MAE_{cv_num}", mae)
                        wrmsses.append(wrmsse)
                        rmses.append(rmse)
                        maes.append(mae)

                mlflow.log_metric("WRMSSE", np.mean(wrmsses))
                mlflow.log_metric("RMSE", np.mean(rmses))
                mlflow.log_metric("MAE", np.mean(maes))
                print("=================================")
                print("WRMSSE", np.mean(wrmsses))
                print("RMSE", np.mean(rmses))
                print("MAE", np.mean(maes))
                print("=================================")


class LGBMSubmission(M5):
    def requires(self):
        return dict(sp=CombineFeatures(), raw=LoadRawData())

    def run(self):
        raw: RawData = self.load("raw")
        sp: Split = self.load("sp")
        sp.train = sp.train[(sp.train.fe_rolling_sum_t0_30 > config.MIN_SUM)][
            config.features + ["sales"]
        ]
        print("train shape:", sp.train.shape)
        print(
            "number of NaN:",
            sp.train.isna().sum(axis=0).where(lambda x: x > 0).dropna().sort_values(),
        )
        # sp.train.dropna(inplace=True)
        # print("train shape:", sp.train.shape)
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

        # for i in range(1, 29):
        #     final["F" + str(i)] *= 1.025

        final.to_csv(submission_file_name, index=False)
        self.dump(model)
