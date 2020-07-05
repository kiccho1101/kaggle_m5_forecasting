import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm.autonotebook import tqdm
import mlflow
import mlflow.lightgbm
import pandas as pd
import time

from kaggle_m5_forecasting.base import Split
from kaggle_m5_forecasting import M5, CombineValFeatures, LoadRawData
from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.config import Config
from kaggle_m5_forecasting.utils import timer

from kaggle_m5_forecasting.task.lgbm import (
    get_run_name,
    start_mlflow,
    drop_outliers,
    delete_unused_features,
    print_nan_ratio,
    log_params,
    convert_to_lgb_dataset,
    get_zero_nonzero_ids,
    partial_train_and_predict,
    train,
    train_by_zero,
    train_by_store,
    train_by_dept,
    train_by_cat,
    predict,
    cls_postprocessing,
    log_result,
    log_metrics,
    log_avg_metrics,
)


class LGBMCrossValidation(M5):
    def requires(self):
        return dict(splits=CombineValFeatures(), raw=LoadRawData())

    def run(self):
        config = Config()

        run_name = get_run_name()

        splits: List[Split] = self.load("splits")
        raw: RawData = self.load("raw")

        splits = delete_unused_features(splits)
        if config.DROP_OUTLIERS:
            splits = drop_outliers(splits)

        print_nan_ratio(splits)

        for SEED in range(1, 1000, 10):

            run_name = "seed = {}".format(SEED)

            experiment_id = start_mlflow()
            mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
            timestamp = mlflow.active_run().info.start_time / 1000
            start_time = datetime.datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d_%H:%M:%S"
            )

            log_params()

            # wrmsses, rmss, maes = [], [], []
            for cv_num, sp in enumerate(splits):
                Path(f"./output/cv/{start_time}/{cv_num}").mkdir(
                    parents=True, exist_ok=True
                )

                test_pred: pd.DataFrame = pd.DataFrame()
                if config.MODEL == "zero":
                    test_pred = train_by_zero(raw, sp, cv_num)
                elif config.MODEL == "store":
                    test_pred = train_by_store(raw, sp, cv_num, SEED)
                elif config.MODEL == "store":
                    test_pred = train_by_store(raw, sp, cv_num)
                elif config.MODEL == "cat":
                    test_pred = train_by_cat(raw, sp, cv_num)
                elif config.MODEL == "dept":
                    test_pred = train_by_dept(raw, sp, cv_num)
                elif config.MODEL == "normal":
                    train_set, val_set = convert_to_lgb_dataset(sp, cv_num)
                    model = train(
                        cv_num,
                        config.lgbm_params,
                        train_set,
                        [train_set],
                        verbose_eval=10,
                        early_stopping_rounds=20,
                    )
                    test_pred = predict(cv_num, sp, model)

                if config.CLS_POSTPROCESSING:
                    wrmsse, rmse, mae = log_metrics(
                        cv_num, start_time, raw, test_pred, sp.test
                    )
                    test_pred = cls_postprocessing(cv_num, test_pred)
                log_result(cv_num, start_time, test_pred)
                # wrmsse, rmse, mae = log_metrics(cv_num, start_time, raw, test_pred, sp.test)
                # wrmsses.append(wrmsse)
                # rmses.append(rmse)
                # maes.append(mae)

            # log_avg_metrics(wrmsses, rmses, maes)
            mlflow.end_run()
            time.sleep(10)
