import pandas as pd
import numpy as np
import time
from typing import List
from contextlib import contextmanager
import mlflow


@contextmanager
def timer(name: str, mlflow_on: bool = False):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.4f} s")
    print()
    if mlflow_on:
        mlflow.log_param(name, f"{time.time() - t0:.4f}s")


def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    if verbose:
        print("START MEM:")
        print_mem_usage(df)
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    if verbose:
        print("END MEM:")
        print_mem_usage(df)

    return df


def print_mem_usage(df: pd.DataFrame):
    print("shape:", df.shape)
    print("Mem Usage: {:.3f} Mb".format(df.memory_usage().sum() / 1024 ** 2))


def merge_by_concat(
    df1: pd.DataFrame, df2: pd.DataFrame, on: List[str]
) -> pd.DataFrame:
    df: pd.DataFrame = df1[on].merge(df2, on=on, how="left")
    new_columns: List[str] = [col for col in list(df) if col not in on]
    df = pd.concat([df1, df[new_columns]], axis=1)
    return df


def logcoshobjective(y_true, y_pred):
    d = y_pred - y_true
    grad = np.tanh(d) / y_true
    hess = (1.0 - grad * grad) / y_true
    return grad, hess


def huberobjective(y_true, y_pred):
    d = y_pred - y_true
    h = 1.2  # h is delta in the formula
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def custom_objective(y_true, y_pred):
    coef = [0.35, 0.5, 0.05, 0.1]

    # fair
    c = 0.5
    residual = y_pred - y_true
    grad = c * residual / (np.abs(residual) + c)
    hess = c ** 2 / (np.abs(residual) + c) ** 2

    # huber
    h = 1.2  # h is delta in the formula
    scale = 1 + (residual / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad_huber = residual / scale_sqrt
    hess_huber = 1 / scale / scale_sqrt

    # rmse grad and hess
    grad_rmse = residual
    hess_rmse = 1.0

    # mae grad and hess
    grad_mae = np.array(residual)
    grad_mae[grad_mae > 0] = 1.0
    grad_mae[grad_mae <= 0] = -1.0
    hess_mae = 1.0

    return (
        coef[0] * grad
        + coef[1] * grad_huber
        + coef[2] * grad_rmse
        + coef[3] * grad_mae,
        coef[0] * hess
        + coef[1] * hess_huber
        + coef[2] * hess_rmse
        + coef[3] * hess_mae,
    )
