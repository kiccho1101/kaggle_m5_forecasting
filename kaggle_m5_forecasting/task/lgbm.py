import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from kaggle_m5_forecasting.base import Split
from kaggle_m5_forecasting.config import Config
from kaggle_m5_forecasting import RawData
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.fe_rolling import make_rolling_for_test
from kaggle_m5_forecasting.cv_result import CVResult
import pickle
import sklearn.metrics
from typing import List, Tuple, Optional
from tqdm.autonotebook import tqdm


def get_run_name() -> str:
    print("please put run_name")
    run_name = input()
    return run_name


def start_mlflow() -> int:
    try:
        mlflow.end_run()
    except Exception:
        pass
    exp_name = "cv_new"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    return experiment_id


def drop_outliers(splits: List[Split]) -> List[Split]:
    for i in range(len(splits)):
        # # Drop all the christmas eve data (12-24)
        # splits[i].train = splits[i].train[
        #     ~splits[i].train["d"].isin([330, 696, 1061, 1426, 1791])
        # ]
        # Drop all the christmas data (12-25)
        splits[i].train = splits[i].train[
            ~splits[i].train["d"].isin([331, 697, 1062, 1427, 1792])
        ]
        # # Drop all  12-26 data
        # splits[i].train = splits[i].train[
        #     ~splits[i].train["d"].isin([331, 697, 1062, 1427, 1792])
        # ]
        # # Drop all  12-27 data
        # splits[i].train = splits[i].train[
        #     ~splits[i].train["d"].isin([332, 698, 1063, 1428, 1793])
        # ]
        # Drop all the thanksgiving data (11-27)
        splits[i].train = splits[i].train[
            ~splits[i].train["d"].isin([300, 664, 1035, 1399, 1413, 1763])
        ]
        # # Drop all the new year data (01-01)
        # splits[i].train = splits[i].train[
        #     ~splits[i].train["d"].isin([338, 704, 1069, 1434, 1799])
        # ]

        print(f"CV{i} outliers dropped train shape:", splits[i].train.shape)
    return splits


def delete_unused_features(splits: List[Split]) -> List[Split]:
    config = Config()
    for i in range(len(splits)):
        splits[i].train = splits[i].train[config.features + [config.TARGET]]
        print(f"CV{i} train shape:", splits[i].train.shape)
        if config.DROP_NA:
            splits[i].train = splits[i].train.dropna()
            print(f"CV{i} NA dropped train shape:", splits[i].train.shape)
    return splits


def log_params():
    config = Config()
    mlflow.lightgbm.autolog()
    mlflow.log_param("MIN_SUM", config.MIN_SUM)
    mlflow.log_param("MAX_LAGS", config.MAX_LAGS)
    mlflow.log_param("start_day", config.START_DAY)
    mlflow.log_param("SEED", config.SEED)
    mlflow.log_param("DROP_NA", config.DROP_NA)
    mlflow.log_param("DROP_OUTLIERS", config.DROP_OUTLIERS)
    mlflow.log_param("CV_SAMPLE_RATE", config.CV_SAMPLE_RATE)
    mlflow.log_param("features", ",\n".join([f"'{f}'" for f in config.features]))


def convert_to_lgb_dataset(sp: Split, cv_num: int) -> Tuple[lgb.Dataset, lgb.Dataset]:
    config = Config()
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
    config = Config()
    with timer(f"train CV_{cv_num}", mlflow_on=True):
        model = lgb.train(
            config.lgbm_params,
            train_set,
            num_boost_round=config.num_boost_round,
            verbose_eval=verbose_eval,
            # early_stopping_rounds=early_stopping_rounds,
            valid_sets=valid_sets,
        )
    return model


def predict(cv_num: int, sp: Split, model: lgb.Booster) -> pd.DataFrame:
    config = Config()
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
    config = Config()
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
    pickle.dump(
        config, open(f"./output/cv/{start_time}/{cv_num}/config.pkl", "wb"),
    )
    test_pred


def log_metrics(
    cv_num: int,
    start_time: str,
    raw: RawData,
    test_pred: pd.DataFrame,
    test_true: pd.DataFrame,
) -> Tuple[float, float, float]:
    config = Config()
    d_start = config.CV_START_DAYS[cv_num]
    d_end = config.CV_START_DAYS[cv_num] + 28

    cv_result = CVResult(
        cv_num=cv_num,
        config=config,
        test_pred=test_pred[(test_pred.d >= d_start) & (test_pred.d < d_end)],
    )
    evaluator = cv_result.get_evaluator(raw)
    cv_result.create_dashboard(raw, f"./output/cv/{start_time}/{cv_num}")
    y_pred = test_pred[(test_pred.d >= d_start) & (test_pred.d < d_end)][config.TARGET]
    y_true = test_true[(test_true.d >= d_start) & (test_true.d < d_end)][config.TARGET]

    wrmsse = np.mean(evaluator.all_scores)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)

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
