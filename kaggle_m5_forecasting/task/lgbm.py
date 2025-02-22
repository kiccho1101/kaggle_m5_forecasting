import lightgbm as lgb
from lightgbm import LGBMClassifier
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
from typing import List, Tuple, Optional, Dict, Any
from tqdm.autonotebook import tqdm


def get_run_name() -> str:
    print("please put run_name")
    run_name = input()
    # run_name = "all data val"
    return run_name


def start_mlflow(exp_name: str = "cv_new") -> int:
    try:
        mlflow.end_run()
    except Exception:
        pass
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    return experiment_id


def delete_unused_features(splits: List[Split]) -> List[Split]:
    config = Config()
    for i in range(len(splits)):
        splits[i].train = splits[i].train[["id", "d", config.TARGET] + config.features]
        splits[i].test = splits[i].test[["id", "d", config.TARGET] + config.features]
        splits[i].train = splits[i].train[splits[i].train["d"] >= config.START_DAY]
        print(f"CV{i} train shape:", splits[i].train.shape)
        if config.DROP_NA:
            splits[i].train = splits[i].train.dropna()
            print(f"CV{i} NA dropped train shape:", splits[i].train.shape)
    return splits


def drop_outliers(splits: List[Split]) -> List[Split]:
    for i in range(len(splits)):
        # # Drop all the christmas data (12-25)
        splits[i].train = splits[i].train[
            ~splits[i].train["d"].isin([331, 697, 1062, 1427, 1792])
        ]
        # Drop all the thanksgiving data (11-27)
        splits[i].train = splits[i].train[
            ~splits[i].train["d"].isin([300, 664, 1035, 1399, 1413, 1763])
        ]
        print(f"CV{i} outliers dropped train shape:", splits[i].train.shape)
    return splits


def print_nan_ratio(splits: List[Split]):
    for cv_num, sp in enumerate(splits):
        nan_ratio = sp.train.isna().sum() / len(sp.train) * 100
        print(f"CV {cv_num} train nan ratio")
        print(nan_ratio[nan_ratio > 0].sort_values(ascending=False).head(25))
        nan_ratio = sp.test.isna().sum() / len(sp.test) * 100
        print(f"CV {cv_num} test nan ratio")
        print(nan_ratio[nan_ratio > 0].sort_values(ascending=False).head(25))


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
    mlflow.log_param("MODEL", config.MODEL)
    mlflow.log_param("CLS_POSTPROCESSING", config.CLS_POSTPROCESSING)
    mlflow.log_param("CLS_TIMESTAMP", config.CLS_TIMESTAMP)
    mlflow.log_param("CLS_THRESHOLD", config.CLS_THRESHOLD)
    mlflow.log_param("features", ",\n".join([f"'{f}'" for f in config.features]))


def get_zero_nonzero_ids(
    raw: RawData, zero_ratio_threshold: float = 0.8
) -> Tuple[pd.Series, pd.Series]:
    raw.sales_train_validation["zero_ratio"] = raw.sales_train_validation.filter(
        like="d_"
    ).apply(lambda row: (row == 0).mean(), axis=1)
    zero_ids = raw.sales_train_validation[
        raw.sales_train_validation["zero_ratio"] > zero_ratio_threshold
    ]["id"]
    nonzero_ids = raw.sales_train_validation[
        raw.sales_train_validation["zero_ratio"] <= zero_ratio_threshold
    ]["id"]
    return zero_ids, nonzero_ids


def convert_to_lgb_dataset(sp: Split, cv_num: int) -> Tuple[lgb.Dataset, lgb.Dataset]:
    config = Config()
    train_set = lgb.Dataset(sp.train[config.features], sp.train[config.TARGET])
    val_set = lgb.Dataset(
        sp.test[
            (sp.test.d >= config.CV_START_DAYS[cv_num])
            & (sp.test.d < config.CV_START_DAYS[cv_num] + 28)
        ][config.features],
        sp.test[
            (sp.test.d >= config.CV_START_DAYS[cv_num])
            & (sp.test.d < config.CV_START_DAYS[cv_num] + 28)
        ][config.TARGET],
    )
    return train_set, val_set


def train(
    cv_num: int,
    params: Dict[str, Any],
    train_set: lgb.Dataset,
    valid_sets: List[lgb.Dataset],
    verbose_eval: int,
    early_stopping_rounds: Optional[int] = None,
    model_number: Optional[int] = None,
) -> lgb.Booster:
    config = Config()
    timer_name: str = f"train CV_{cv_num}"
    if model_number:
        timer_name += f"_{model_number}"
    with timer(timer_name, mlflow_on=True):
        model = lgb.train(
            params,
            train_set,
            num_boost_round=config.num_boost_round,
            verbose_eval=verbose_eval,
            # early_stopping_rounds=early_stopping_rounds,
            valid_sets=valid_sets,
        )
    return model


def train_by_zero(raw: RawData, sp: Split, cv_num: int) -> pd.DataFrame:
    zero_ids, nonzero_ids = get_zero_nonzero_ids(raw, 0.9)
    test_pred_zero = partial_train_and_predict(sp, zero_ids, cv_num, 0)
    test_pred_nonzero = partial_train_and_predict(sp, nonzero_ids, cv_num, 1)
    test_pred = pd.concat([test_pred_zero, test_pred_nonzero], axis=0)
    return test_pred


def train_by_store(
    raw: RawData, sp: Split, cv_num: int, SEED: Optional[int] = None
) -> pd.DataFrame:
    test_pred = pd.DataFrame()
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
        test_pred_part = partial_train_and_predict(sp, ids, cv_num, i, SEED=SEED)
        test_pred = pd.concat([test_pred, test_pred_part], axis=0)
    return test_pred


def train_by_dept(raw: RawData, sp: Split, cv_num: int) -> pd.DataFrame:
    test_pred = pd.DataFrame()
    for i, (dept_id, ids) in enumerate(
        [
            (
                store_id,
                raw.sales_train_validation[
                    raw.sales_train_validation["dept_id"] == store_id
                ]["id"],
            )
            for store_id in raw.sales_train_validation["dept_id"].unique()
        ]
    ):
        print(dept_id)
        test_pred_part = partial_train_and_predict(sp, ids, cv_num, i)
        test_pred = pd.concat([test_pred, test_pred_part], axis=0)
    return test_pred


def train_by_cat(raw: RawData, sp: Split, cv_num: int) -> pd.DataFrame:
    test_pred = pd.DataFrame()
    for i, (cat_id, ids) in enumerate(
        [
            (
                store_id,
                raw.sales_train_validation[
                    raw.sales_train_validation["cat_id"] == store_id
                ]["id"],
            )
            for store_id in raw.sales_train_validation["cat_id"].unique()
        ]
    ):
        print(cat_id)
        test_pred_part = partial_train_and_predict(sp, ids, cv_num, i)
        test_pred = pd.concat([test_pred, test_pred_part], axis=0)
    return test_pred


def train_cls(
    cv_num: int,
    params: Dict[str, Any],
    train_set: lgb.Dataset,
    valid_sets: List[lgb.Dataset],
    verbose_eval: int,
    early_stopping_rounds: Optional[int] = None,
    model_number: Optional[int] = None,
) -> LGBMClassifier:
    config = Config()
    timer_name: str = f"train CV_{cv_num}"
    if model_number:
        timer_name += f"_{model_number}"
    with timer(timer_name, mlflow_on=True):
        model = LGBMClassifier(**config.lgbm_cls_params)
        model.fit(
            train_set.data,
            train_set.label,
            categorical_feature=config.lgbm_cat_features,
            eval_set=[(dataset.data, dataset.label) for dataset in valid_sets],
            eval_metric="logloss,auc,cross_entropy",
            verbose=10,
        )
    return model


def predict(
    cv_num: int, sp: Split, model: lgb.Booster, model_number: Optional[int] = None
) -> pd.DataFrame:
    config = Config()
    d_start: int = config.CV_START_DAYS[cv_num]
    d_end: int = config.CV_START_DAYS[cv_num] + 28
    test_pred = sp.test.copy()
    test_pred[config.TARGET + "_true"] = test_pred[config.TARGET]

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


def predict_cls(
    sp: Split, cv_num: int, model: LGBMClassifier, val_set: lgb.Dataset
) -> pd.DataFrame:
    config = Config()
    df_val = sp.test[
        (sp.test.d >= config.CV_START_DAYS[cv_num])
        & (sp.test.d < config.CV_START_DAYS[cv_num] + 28)
    ][["id", "d", "sales_is_zero"]]
    df_val["sales_is_zero_pred"] = model.predict_proba(val_set.data)[:, 1]
    return df_val


def cls_postprocessing(cv_num: int, test_pred: pd.DataFrame) -> pd.DataFrame:
    with timer("cls_postprocessing"):
        config = Config()
        df_val: pd.dataframe = pickle.load(
            open(f"./output/cv_cls/{config.CLS_TIMESTAMP}/0/df_val.pkl", "rb")
        )
        test_pred["tmp_id"] = (
            test_pred["id"].astype(str) + "_" + test_pred["d"].astype(str)
        )
        df_val = df_val[df_val["sales_is_zero_pred"] >= config.CLS_THRESHOLD]
        tmp_ids = df_val["id"].astype(str) + "_" + df_val["d"].astype(str)
        test_pred.loc[test_pred["tmp_id"].isin(tmp_ids), "sales"] = 0
        test_pred.drop(["tmp_id"], axis=1, inplace=True)
    return test_pred


def partial_train_and_predict(
    sp: Split,
    ids: pd.Series,
    cv_num: int,
    model_number: int,
    objective: Optional[str] = None,
    SEED: Optional[int] = None,
) -> pd.DataFrame:
    config = Config()
    sp_part: Split = Split()
    sp_part.train = sp.train[sp.train["id"].isin(ids)]
    sp_part.test = sp.test[sp.test["id"].isin(ids)]
    train_set, val_set = convert_to_lgb_dataset(sp_part, cv_num)
    params = config.lgbm_params
    if objective:
        params["objective"] = objective
        params.pop("tweedie_variance_power", None)
    if SEED:
        params["seed"] = SEED
    model = train(
        cv_num, params, train_set, [train_set], 10, 20, model_number=model_number,
    )
    test_pred = predict(cv_num, sp_part, model)
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
