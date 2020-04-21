import pandas as pd
import numpy as np
import time
from typing import List, Any
from contextlib import contextmanager
import mlflow
import psutil
from multiprocessing import Pool


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


def df_parallelize_run(func, t_split: List[Any]):
    num_cores = np.min([psutil.cpu_count(), len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df
