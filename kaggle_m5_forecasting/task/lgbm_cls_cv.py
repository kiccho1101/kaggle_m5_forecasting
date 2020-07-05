import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm.autonotebook import tqdm
import mlflow
import mlflow.lightgbm
import pandas as pd
import pickle

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
    train_cls,
    predict_cls,
)


class LGBMClassifierCrossValidation(M5):
    def requires(self):
        return dict(splits=CombineValFeatures())

    def run(self):
        config = Config()

        run_name = get_run_name()

        splits: List[Split] = self.load("splits")

        splits = delete_unused_features(splits)
        if config.DROP_OUTLIERS:
            splits = drop_outliers(splits)

        experiment_id = start_mlflow("cv_cls")
        mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        timestamp = mlflow.active_run().info.start_time / 1000
        start_time = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d_%H:%M:%S"
        )

        log_params()

        for cv_num, sp in enumerate(splits):
            file_dir = f"./output/cv_cls/{start_time}/{cv_num}"
            Path(file_dir).mkdir(parents=True, exist_ok=True)

            train_set, val_set = convert_to_lgb_dataset(sp, cv_num)
            model = train_cls(
                cv_num, config.lgbm_cls_params, train_set, [val_set], 10, 20
            )
            df_val = predict_cls(sp, cv_num, model, val_set)

            pickle.dump(model, open(f"{file_dir}/model.pkl", "wb"))
            pickle.dump(
                df_val, open(f"{file_dir}/df_val.pkl", "wb"),
            )
        mlflow.end_run()
