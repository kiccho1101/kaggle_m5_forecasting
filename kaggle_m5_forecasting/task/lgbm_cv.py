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
            if mlflow.get_experiment_by_name("cv_classification") is None:
                mlflow.create_experiment("cv_classification")

            wrmsses, rmses, maes = [], [], []
            accuracies, pr_aucs, precisions, recalls = [], [], [], []
            if config.TARGET == "sales":
                experiment_id = mlflow.get_experiment_by_name("cv").experiment_id
            elif config.TARGET == "sales_is_zero":
                experiment_id = mlflow.get_experiment_by_name(
                    "cv_classification"
                ).experiment_id
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
                            ][config.features + [config.TARGET]]
                            print("train shape:", sp.train.shape)
                            if config.DROP_NA:
                                sp.train.dropna(inplace=True)
                                print("train shape:", sp.train.shape)
                            mlflow.log_param(f"train shape CV_{cv_num}", sp.train.shape)
                            train_set = lgb.Dataset(
                                sp.train[config.features], sp.train[config.TARGET]
                            )
                            val_set = lgb.Dataset(
                                sp.test[sp.test.d > config.CV_START_DAYS[cv_num]][
                                    config.features
                                ],
                                sp.test[sp.test.d > config.CV_START_DAYS[cv_num]][
                                    config.TARGET
                                ],
                            )

                        with timer(f"lgbm train CV_{cv_num}", mlflow_on=True):
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
                                    early_stopping_rounds=20,
                                    verbose=50,
                                    categorical_feature=config.lgbm_cat_features,
                                    eval_set=(
                                        sp.test[
                                            sp.test.d > config.CV_START_DAYS[cv_num]
                                        ][config.features],
                                        sp.test[
                                            sp.test.d > config.CV_START_DAYS[cv_num]
                                        ][config.TARGET],
                                    ),
                                )
                            if config.TARGET == "sales":
                                model = lgb.train(
                                    params,
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
                            test_pred.loc[
                                test_pred.d >= config.CV_START_DAYS[cv_num],
                                config.TARGET + "_true",
                            ] = test_pred.loc[
                                test_pred.d >= config.CV_START_DAYS[cv_num],
                                config.TARGET,
                            ]
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
                                    test_pred.d == d, config.TARGET
                                ] = model.predict(
                                    test_pred.loc[test_pred.d == d, config.features]
                                )
                                if config.TARGET == "sales_is_zero":
                                    test_pred.loc[
                                        test_pred.d == d, config.TARGET + "_proba"
                                    ] = model.predict_proba(
                                        test_pred.loc[test_pred.d == d, config.features]
                                    )[
                                        :, 1
                                    ]
                                if config.TARGET == "sales":
                                    test_pred.loc[test_pred.d == d, "sales_is_zero"] = (
                                        test_pred.loc[test_pred.d == d] == 0
                                    ).astype(np.int8)
                            if config.TARGET == "sales":
                                print(
                                    test_pred.loc[
                                        (test_pred.d >= config.CV_START_DAYS[cv_num])
                                        & (
                                            test_pred.d
                                            <= config.CV_START_DAYS[cv_num] + 28
                                        ),
                                        config.TARGET,
                                    ].values.mean()
                                )
                                cls_pred: pd.DataFrame = pickle.load(
                                    open(f"cv_pred_{cv_num}.pkl", "rb")
                                )
                                test_pred.loc[
                                    (
                                        cls_pred["sales_is_zero_proba"]
                                        >= config.CLS_THRESHOLD
                                    ).index,
                                    "sales",
                                ] = 0
                                print(
                                    test_pred.loc[
                                        (test_pred.d >= config.CV_START_DAYS[cv_num])
                                        & (
                                            test_pred.d
                                            <= config.CV_START_DAYS[cv_num] + 28
                                        ),
                                        config.TARGET,
                                    ].values.mean()
                                )
                            if config.TARGET == "sales_is_zero":
                                pickle.dump(
                                    test_pred, open(f"cv_pred_{cv_num}.pkl", "wb")
                                )

                        with timer(f"calc metrics CV_{cv_num}", mlflow_on=True):
                            if config.TARGET == "sales":
                                wrmsse, rmse, mae, evaluator = calc_metrics(
                                    raw,
                                    test_pred[
                                        (test_pred.d >= config.CV_START_DAYS[cv_num])
                                        & (
                                            test_pred.d
                                            < config.CV_START_DAYS[cv_num] + 28
                                        )
                                    ],
                                    test_true[
                                        (test_true.d >= config.CV_START_DAYS[cv_num])
                                        & (
                                            test_true.d
                                            < config.CV_START_DAYS[cv_num] + 28
                                        )
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
                            elif config.TARGET == "sales_is_zero":
                                y_true = test_true.loc[
                                    (test_true.d >= config.CV_START_DAYS[cv_num])
                                    & (test_true.d < config.CV_START_DAYS[cv_num] + 28),
                                    config.TARGET,
                                ].values
                                y_pred = test_pred.loc[
                                    (test_pred.d >= config.CV_START_DAYS[cv_num])
                                    & (test_pred.d < config.CV_START_DAYS[cv_num] + 28),
                                    config.TARGET,
                                ].values
                                accuracy = sklearn.metrics.accuracy_score(
                                    y_true, y_pred
                                )
                                accuracies.append(accuracy)
                                precision = sklearn.metrics.precision_score(
                                    y_true, y_pred
                                )
                                recall = sklearn.metrics.recall_score(y_true, y_pred)
                                precisions.append(precision)
                                recalls.append(recall)
                                mlflow.log_metric(f"accuracy_{cv_num}", accuracy)
                                mlflow.log_metric(f"precision_{cv_num}", precision)
                                mlflow.log_metric(f"recall_{cv_num}", recall)
                                print(f"==========CV No: {cv_num}=================")
                                print("accuracy", accuracy)
                                print("precision", precision)
                                print("recall", recall)
                                print("=================================")

                if config.TARGET == "sales":
                    mlflow.log_metric("WRMSSE", np.mean(wrmsses))
                    mlflow.log_metric("RMSE", np.mean(rmses))
                    mlflow.log_metric("MAE", np.mean(maes))
                    print("=================================")
                    print("WRMSSE", np.mean(wrmsses))
                    print("RMSE", np.mean(rmses))
                    print("MAE", np.mean(maes))
                    print("=================================")
                elif config.TARGET == "sales_is_zero":
                    mlflow.log_metric("accuracy", np.mean(accuracies))
                    mlflow.log_metric("prAUC", np.mean(pr_aucs))
                    print("=================================")
                    print("accuracy", np.mean(accuracies))
                    print("prAUC", np.mean(pr_aucs))
                    print("precision", np.mean(precisions))
                    print("recall", np.mean(recalls))
                    print("=================================")
