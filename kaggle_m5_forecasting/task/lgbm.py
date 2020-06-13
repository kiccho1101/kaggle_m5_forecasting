import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from kaggle_m5_forecasting.base import Split
from kaggle_m5_forecasting import RawData, config
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.fe_rolling import make_rolling_for_test
from kaggle_m5_forecasting.metric import calc_metrics
import pickle
from typing import List, Tuple, Optional
from tqdm.autonotebook import tqdm


def start_mlflow() -> int:
    try:
        mlflow.end_run()
    except Exception:
        pass
    if mlflow.get_experiment_by_name("cv") is None:
        mlflow.create_experiment("cv")
    experiment_id = mlflow.get_experiment_by_name("cv").experiment_id
    return experiment_id


def delete_unused_features(splits: List[Split]) -> List[Split]:
    for i in range(len(splits)):
        splits[i].train = splits[i].train[config.features + [config.TARGET]]
        print(f"CV{i} train shape:", splits[i].train.shape)
        if config.DROP_NA:
            splits[i].train = splits[i].train.dropna()
            print(f"CV{i} dropped train shape:", splits[i].train.shape)
    return splits


def log_params():
    mlflow.lightgbm.autolog()
    mlflow.log_param("MIN_SUM", config.MIN_SUM)
    mlflow.log_param("MAX_LAGS", config.MAX_LAGS)
    mlflow.log_param("start_day", config.START_DAY)
    mlflow.log_param("SEED", config.SEED)
    mlflow.log_param("features", ",\n".join([f"'{f}'" for f in config.features]))


def convert_to_lgb_dataset(sp: Split, cv_num: int) -> Tuple[lgb.Dataset, lgb.Dataset]:
    train_set = lgb.Dataset(sp.train[config.features], sp.train[config.TARGET])
    val_set = lgb.Dataset(
        sp.test[sp.test.d > config.CV_START_DAYS[cv_num]][config.features],
        sp.test[sp.test.d > config.CV_START_DAYS[cv_num]][config.TARGET],
    )
    return train_set, val_set


def train(
    cv_num: int,
    train_set: lgb.Dataset,
    valid_sets: List[lgb.Dataset],
    verbose_eval: int,
    early_stopping_rounds: Optional[int] = None,
) -> lgb.Booster:
    with timer(f"train CV_{cv_num}", mlflow_on=True):
        model = lgb.train(
            config.lgbm_params,
            train_set,
            num_boost_round=config.num_boost_round,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            valid_sets=valid_sets,
        )
    return model


def predict(cv_num: int, sp: Split, model: lgb.Booster) -> pd.DataFrame:
    d_start: int = config.CV_START_DAYS[cv_num]
    d_end: int = config.CV_START_DAYS[cv_num] + 28
    test_pred = sp.test.copy()
    test_pred[config.TARGET + "_true"] = test_pred[config.TARGET]

    with timer(f"predict CV_{cv_num}", mlflow_on=True):
        test_pred.loc[test_pred.d >= d_start, config.TARGET] = np.nan
        for d in tqdm(range(d_start, d_end)):
            test_pred = make_rolling_for_test(test_pred, d, config.features)
            test_pred.loc[test_pred.d == d, config.TARGET] = model.predict(
                test_pred.loc[test_pred.d == d, config.features]
            )
            test_pred.loc[test_pred.d == d, "sales_is_zero"] = (
                test_pred.loc[test_pred.d == d, "sales"] == 0
            ).astype(np.int8)

    return test_pred


def log_result(cv_num: int, start_time: str, test_pred: pd.DataFrame):
    d_start = config.CV_START_DAYS[cv_num]
    d_end = config.CV_START_DAYS[cv_num] + 28
    save_cols: List[str] = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        config.TARGET,
        config.TARGET + "_true",
    ]
    pickle.dump(
        test_pred.loc[(test_pred.d >= d_start) & (test_pred.d < d_end), save_cols],
        open(f"./output/cv/{start_time}/{cv_num}/test_pred.pkl", "wb"),
    )
    test_pred


def log_metrics(
    cv_num: int,
    start_time: str,
    raw: RawData,
    test_pred: pd.DataFrame,
    test_true: pd.DataFrame,
) -> Tuple[float, float, float]:
    d_start = config.CV_START_DAYS[cv_num]
    d_end = config.CV_START_DAYS[cv_num] + 28
    wrmsse, rmse, mae, _ = calc_metrics(
        raw,
        test_pred[(test_pred.d >= d_start) & (test_pred.d < d_end)],
        test_true[(test_true.d >= d_start) & (test_true.d < d_end)],
    )
    print(f"==========CV No: {cv_num}=================")
    print("WRMSSE", wrmsse)
    print("RMSE", rmse)
    print("MAE", mae)
    print("=================================")
    mlflow.log_metric(f"WRMSSE_{cv_num}", wrmsse)
    mlflow.log_metric(f"RMSE_{cv_num}", rmse)
    mlflow.log_metric(f"MAE_{cv_num}", mae)
    return wrmsse, rmse, mae


def log_avg_metrics(wrmsses: List[float], rmses: List[float], maes: List[float]):
    mlflow.log_metric("WRMSSE", np.mean(wrmsses))
    mlflow.log_metric("RMSE", np.mean(rmses))
    mlflow.log_metric("MAE", np.mean(maes))
    print("=================================")
    print("WRMSSE", np.mean(wrmsses))
    print("RMSE", np.mean(rmses))
    print("MAE", np.mean(maes))
    print("=================================")
