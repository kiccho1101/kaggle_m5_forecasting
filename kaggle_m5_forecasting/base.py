import gokart
from typing import List
import pandas as pd


class M5(gokart.TaskOnKart):
    task_namespace = "m5"


class SplitIndex:
    train: List[int]
    test: List[int]


class Split:
    train: pd.DataFrame
    test: pd.DataFrame
