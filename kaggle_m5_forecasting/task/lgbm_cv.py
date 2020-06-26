import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm.autonotebook import tqdm
import mlflow
import mlflow.lightgbm
import pandas as pd

from kaggle_m5_forecasting.base import Split
from kaggle_m5_forecasting import M5, CombineValFeatures, LoadRawData
from kaggle_m5_forecasting.data.load_data import RawData
from kaggle_m5_forecasting.config import Config

from kaggle_m5_forecasting.task.lgbm import (
    get_run_name,
    start_mlflow,
    drop_outliers,
    delete_unused_features,
    log_params,
    convert_to_lgb_dataset,
    get_zero_nonzero_ids,
    partial_train_and_predict,
    train,
    predict,
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

        experiment_id = start_mlflow()
        mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        timestamp = mlflow.active_run().info.start_time / 1000
        start_time = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d_%H:%M:%S"
        )

        log_params()

        wrmsses, rmses, maes = [], [], []
        for cv_num, sp in enumerate(splits):
            Path(f"./output/cv/{start_time}/{cv_num}").mkdir(
                parents=True, exist_ok=True
            )

            test_pred: pd.DataFrame = pd.DataFrame()
            if config.MODEL == "zero":
                zero_ids, nonzero_ids = get_zero_nonzero_ids(raw, 0.9)
                test_pred_zero = partial_train_and_predict(sp, zero_ids, cv_num, 0)
                test_pred_nonzero = partial_train_and_predict(
                    sp, nonzero_ids, cv_num, 1
                )
                test_pred = pd.concat([test_pred_zero, test_pred_nonzero], axis=0)
            elif config.MODEL == "store":
                for i, (store_id, ids) in enumerate(
                    [
                        (
                            store_id,
                            raw.sales_train_validation[
                                raw.sales_train_validation["store_id"] == store_id
                            ]["id"],
                        )
                        for store_id in raw.sales_train_validation["store_id"].unique()
                    ]
                ):
                    print(store_id)
                    test_pred_part = partial_train_and_predict(sp, ids, cv_num, i)
                    test_pred = pd.concat([test_pred, test_pred_part], axis=0)
            elif config.MODEL == "normal":
                train_set, val_set = convert_to_lgb_dataset(sp, cv_num)
                model = train(cv_num, config.lgbm_params, train_set, [val_set], 10, 20)
                test_pred = predict(cv_num, sp, model)

            log_result(cv_num, start_time, test_pred)
            wrmsse, rmse, mae = log_metrics(cv_num, start_time, raw, test_pred, sp.test)
            wrmsses.append(wrmsse)
            rmses.append(rmse)
            maes.append(mae)

        log_avg_metrics(wrmsses, rmses, maes)
        mlflow.end_run()
