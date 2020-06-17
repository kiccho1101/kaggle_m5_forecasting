import numpy as np

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle_m5_forecasting.wrmsse import WRMSSEEvaluator

from kaggle_m5_forecasting.data.load_data import RawData


def create_viz_df(df: pd.DataFrame, lv: int, raw: RawData):

    df = df.T.reset_index()
    if lv in [6, 7, 8, 9, 11, 12]:
        df.columns = [
            i[0] + "_" + i[1] if i != ("index", "") else i[0] for i in df.columns
        ]
    df = df.merge(
        raw.calendar.loc[:, ["d", "date"]], how="left", left_on="index", right_on="d"
    )
    df["date"] = pd.to_datetime(df.date)
    df = df.set_index("date")
    df = df.drop(["index", "d"], axis=1)

    return df


def create_dashboard(evaluator: WRMSSEEvaluator, raw: RawData):

    wrmsses = [np.mean(evaluator.all_scores)] + evaluator.all_scores
    labels = ["Overall"] + [f"Level {i}" for i in range(1, 13)]

    # WRMSSE by Level
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x=labels, y=wrmsses)
    ax.set(xlabel="", ylabel="WRMSSE")
    plt.title("WRMSSE by Level", fontsize=20, fontweight="bold")
    for index, val in enumerate(wrmsses):
        ax.text(index * 1, val + 0.01, round(val, 4), color="black", ha="center")

    # configuration array for the charts
    n_rows = [1, 1, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3]
    n_cols = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    width = [7, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
    height = [4, 3, 12, 3, 9, 9, 9, 9, 9, 9, 9, 9]

    for i in range(1, 13):

        scores = getattr(evaluator, f"lv{i}_scores")
        weights = getattr(evaluator, f"lv{i}_weight")

        if i > 1 and i < 9:
            if i < 7:
                fig, axs = plt.subplots(1, 2, figsize=(12, 3))
            else:
                fig, axs = plt.subplots(2, 1, figsize=(12, 8))

            # RMSSE plot
            scores.plot.bar(width=0.8, ax=axs[0], color="g")
            axs[0].set_title("RMSSE", size=14)
            axs[0].set(xlabel="", ylabel="RMSSE")
            if i >= 4:
                axs[0].tick_params(labelsize=8)
            for index, val in enumerate(scores):
                axs[0].text(
                    index * 1,
                    val + 0.01,
                    round(val, 4),
                    color="black",
                    ha="center",
                    fontsize=10 if i == 2 else 8,
                )

            # Weight plot
            weights.plot.bar(width=0.8, ax=axs[1])
            axs[1].set_title("Weight", size=14)
            axs[1].set(xlabel="", ylabel="Weight")
            if i >= 4:
                axs[1].tick_params(labelsize=8)
            for index, val in enumerate(weights):
                axs[1].text(
                    index * 1,
                    val + 0.01,
                    round(val, 2),
                    color="black",
                    ha="center",
                    fontsize=10 if i == 2 else 8,
                )

            fig.suptitle(
                f"Level {i}: {evaluator.group_ids[i-1]}",
                size=24,
                y=1.1,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.show()

        trn = create_viz_df(
            getattr(evaluator, f"lv{i}_train_df").iloc[:, -28 * 3 :], i, raw
        )
        val = create_viz_df(getattr(evaluator, f"lv{i}_valid_df"), i, raw)
        pred = create_viz_df(getattr(evaluator, f"lv{i}_valid_preds"), i, raw)

        n_cate = trn.shape[1] if i < 7 else 9

        fig, axs = plt.subplots(
            n_rows[i - 1], n_cols[i - 1], figsize=(width[i - 1], height[i - 1])
        )
        if i > 1:
            axs = axs.flatten()

        # Time series plot
        for k in range(0, n_cate):

            ax = axs[k] if i > 1 else axs

            trn.iloc[:, k].plot(ax=ax, label="train")
            val.iloc[:, k].plot(ax=ax, label="valid")
            pred.iloc[:, k].plot(ax=ax, label="pred")
            ax.set_title(f"{trn.columns[k]}  RMSSE:{scores[k]:.4f}", size=14)
            ax.set(xlabel="", ylabel="sales")
            ax.tick_params(labelsize=8)
            ax.legend(loc="upper left", prop={"size": 10})

        if i == 1 or i >= 9:
            fig.suptitle(
                f"Level {i}: {evaluator.group_ids[i-1]}",
                size=24,
                y=1.1,
                fontweight="bold",
            )
        plt.tight_layout()
        plt.show()
