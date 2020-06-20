import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm.autonotebook import tqdm
import mlflow
import mlflow.lightgbm

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

        if config.DROP_OUTLIERS:
            splits = drop_outliers(splits)
        splits = delete_unused_features(splits)

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

            train_set, val_set = convert_to_lgb_dataset(sp, cv_num)
            model = train(cv_num, train_set, [val_set], 10, 20)
            test_pred = predict(cv_num, sp, model)

            log_result(cv_num, start_time, test_pred)
            wrmsse, rmse, mae = log_metrics(cv_num, start_time, raw, test_pred, sp.test)
            wrmsses.append(wrmsse)
            rmses.append(rmse)
            maes.append(mae)

        log_avg_metrics(wrmsses, rmses, maes)
        mlflow.end_run()
