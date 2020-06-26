# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import catch22
import seaborn as sns
from thunderbolt import Thunderbolt
import scipy
import sklearn.preprocessing
import sklearn.cluster

import sys
import os

sys.path.append(os.getcwd() + "/../..")
from kaggle_m5_forecasting.utils import timer

tb = Thunderbolt("./../../resource")
data: pd.DataFrame = tb.get_data("MakeData")
data = data[data.d < 1942]

# %%


with timer("calc grouped aggregates"):
    grouped = data.groupby(["id"])["sales"].agg(
        {
            "mean": lambda x: x.dropna().values.mean(),
            "percentile25": lambda x: x.dropna()
            .sort_values()[: int(len(x) * 0.25)]
            .mean(),
            "percentile50": lambda x: x.dropna()
            .sort_values()[int(len(x) * 0.25) : int(len(x) * 0.5)]
            .mean(),
            "percentile75": lambda x: x.dropna()
            .sort_values()[int(len(x) * 0.5) : int(len(x) * 0.75)]
            .mean(),
            "percentile100": lambda x: x.dropna()
            .sort_values()[int(len(x) * 0.75) :]
            .mean(),
            "std": lambda x: x.dropna().values.std(),
            "CO_Embed2_Dist_tau_d_expfit_meandiff": lambda x: catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(
                x.dropna().tolist()
            ),
            "CO_f1ecac": lambda x: catch22.CO_f1ecac(x.dropna().tolist()),
            "CO_FirstMin_ac": lambda x: catch22.CO_FirstMin_ac(x.dropna().tolist()),
            "CO_HistogramAMI_even_2_5": lambda x: catch22.CO_HistogramAMI_even_2_5(
                x.dropna().tolist()
            ),
            "CO_trev_1_num": lambda x: catch22.CO_trev_1_num(x.dropna().tolist()),
            "DN_HistogramMode_10": lambda x: catch22.DN_HistogramMode_10(
                x.dropna().tolist()
            ),
            "DN_HistogramMode_5": lambda x: catch22.DN_HistogramMode_5(
                x.dropna().tolist()
            ),
            "DN_OutlierInclude_n_001_mdrmd": lambda x: catch22.DN_OutlierInclude_n_001_mdrmd(
                x.dropna().tolist()
            ),
            "DN_OutlierInclude_p_001_mdrmd": lambda x: catch22.DN_OutlierInclude_p_001_mdrmd(
                x.dropna().tolist()
            ),
            "FC_LocalSimple_mean1_tauresrat": lambda x: catch22.FC_LocalSimple_mean1_tauresrat(
                x.dropna().tolist()
            ),
            "FC_LocalSimple_mean3_stderr": lambda x: catch22.FC_LocalSimple_mean3_stderr(
                x.dropna().tolist()
            ),
            "IN_AutoMutualInfoStats_40_gaussian_fmmi": lambda x: catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(
                x.dropna().tolist()
            ),
            "MD_hrv_classic_pnn40": lambda x: catch22.MD_hrv_classic_pnn40(
                x.dropna().tolist()
            ),
            "PD_PeriodicityWang_th0_01": lambda x: catch22.PD_PeriodicityWang_th0_01(
                x.dropna().tolist()
            ),
            "SB_BinaryStats_diff_longstretch0": lambda x: catch22.SB_BinaryStats_diff_longstretch0(
                x.dropna().tolist()
            ),
            "SB_BinaryStats_mean_longstretch1": lambda x: catch22.SB_BinaryStats_mean_longstretch1(
                x.dropna().tolist()
            ),
            "SB_MotifThree_quantile_hh": lambda x: catch22.SB_MotifThree_quantile_hh(
                x.dropna().tolist()
            ),
            "SB_TransitionMatrix_3ac_sumdiagcov": lambda x: catch22.SB_TransitionMatrix_3ac_sumdiagcov(
                x.dropna().tolist()
            ),
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": lambda x: catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(
                x.dropna().tolist()
            ),
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": lambda x: catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(
                x.dropna().tolist()
            ),
            "SP_Summaries_welch_rect_area_5_1": lambda x: catch22.SP_Summaries_welch_rect_area_5_1(
                x.dropna().tolist()
            ),
            "SP_Summaries_welch_rect_centroid": lambda x: catch22.SP_Summaries_welch_rect_centroid(
                x.dropna().tolist()
            ),
        }
    )
grouped["std"] = data.groupby("id")["sales"].agg({"std": "std"})["std"]
grouped["sell_price_mean"] = data.groupby("id")["sell_price"].mean()
grouped = pd.concat(
    [
        grouped,
        pd.get_dummies(
            grouped.reset_index("id")["id"].map(lambda x: "_".join(x.split("_")[:1])),
            prefix="genre",
        ).set_index(grouped.index),
    ],
    axis=1,
)
grouped = pd.concat(
    [
        grouped,
        pd.get_dummies(
            grouped.reset_index("id")["id"].map(lambda x: "_".join(x.split("_")[:2])),
            prefix="genre",
        ).set_index(grouped.index),
    ],
    axis=1,
)

# %%

scaled = sklearn.preprocessing.StandardScaler().fit_transform(
    grouped[
        [
            "mean",
            "percentile25",
            "percentile50",
            "std",
            "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "CO_f1ecac",
            "CO_FirstMin_ac",
            "CO_HistogramAMI_even_2_5",
            "CO_trev_1_num",
            "DN_HistogramMode_10",
            "DN_HistogramMode_5",
            "DN_OutlierInclude_n_001_mdrmd",
            "DN_OutlierInclude_p_001_mdrmd",
            "FC_LocalSimple_mean1_tauresrat",
            "FC_LocalSimple_mean3_stderr",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi",
            "MD_hrv_classic_pnn40",
            "PD_PeriodicityWang_th0_01",
            "SB_BinaryStats_diff_longstretch0",
            "SB_BinaryStats_mean_longstretch1",
            "SB_MotifThree_quantile_hh",
            "SB_TransitionMatrix_3ac_sumdiagcov",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SP_Summaries_welch_rect_area_5_1",
            "SP_Summaries_welch_rect_centroid",
            "sell_price_mean",
            # "genre_FOODS",
            # "genre_HOBBIES",
            # "genre_HOUSEHOLD",
            "genre_FOODS_1",
            "genre_FOODS_2",
            "genre_FOODS_3",
            "genre_HOBBIES_1",
            "genre_HOBBIES_2",
            "genre_HOUSEHOLD_1",
            "genre_HOUSEHOLD_2",
        ]
    ].values
)
grouped["fe_cluster"] = sklearn.cluster.KMeans(n_clusters=3).fit(scaled).labels_
grouped["fe_cluster"].value_counts()


# %%
for i in grouped[grouped.fe_cluster == 3].sample(20).index.tolist():
    print(i)
    d = data[data.id == i]
    d["sales"].plot()
    plt.show()


# %%
grouped.filter(like="genre")
