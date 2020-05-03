# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from thunderbolt import Thunderbolt

tb = Thunderbolt("./resource")
data: pd.DataFrame = tb.get_data("MakeData")


# %%

df = data[data.id == data.loc[np.random.randint(0, len(data) - 1), "id"]].reset_index(
    drop=True
)

lag = 7
w_sizes = [7, 30, 60, 90, 180]
for w_size in w_sizes:
    df[f"r_t{lag}_{w_size}"] = df["sales"].shift(lag).rolling(w_size).mean()
    df[f"sub_t{lag}_{w_size}"] = (
        df["sales"].shift(lag) - df["sales"].shift(lag + 7)
    ).rolling(w_size).mean() + df["sales"].shift(lag).rolling(w_size).mean()
    df.loc[df[f"sub_t{lag}_{w_size}"] < 0, f"sub_t{lag}_{w_size}"] = 0
features = (
    ["sales"]
    + [f"r_t{lag}_{w_size}" for w_size in w_sizes]
    + [f"sub_t{lag}_{w_size}" for w_size in w_sizes]
)
df[features].plot()
plt.show()


df = df[features].dropna()
pd.concat(
    [
        df.drop("sales", axis=1)
        .apply(lambda x: np.sqrt(sklearn.metrics.mean_squared_error(df["sales"], x)))
        .rename("rmse"),
        df.drop("sales", axis=1)
        .apply(lambda x: sklearn.metrics.mean_absolute_error(df["sales"], x))
        .rename("mae"),
    ],
    axis=1,
).sort_values("mae")

# %%
df = data[data.id == data.loc[np.random.randint(0, len(data) - 1), "id"]].reset_index(
    drop=True
)

# %%

# %%
import catch22
import seaborn as sns

df = data[data.id == data.loc[np.random.randint(0, len(data) - 1), "id"]]
df["sales"].plot()
plt.show()

sns.distplot(df["sales"])
plt.show()


pd.Series(
    {
        "mean": df["sales"].dropna().mean(),
        "zero_ratio": df["sales_is_zero"].dropna().mean(),
        "std": df["sales"].dropna().std(),
        "kurt": df["sales"].dropna().kurt(),
        "skew": df["sales"].dropna().skew(),
        "CO_Embed2_Dist_tau_d_expfit_meandiff": catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(
            df["sales"].dropna().tolist()
        ),
        "CO_f1ecac": catch22.CO_f1ecac(df["sales"].dropna().tolist()),
        "CO_FirstMin_ac": catch22.CO_FirstMin_ac(df["sales"].dropna().tolist()),
        "CO_HistogramAMI_even_2_5": catch22.CO_HistogramAMI_even_2_5(
            df["sales"].dropna().tolist()
        ),
        "CO_trev_1_num": catch22.CO_trev_1_num(df["sales"].dropna().tolist()),
        "DN_HistogramMode_10": catch22.DN_HistogramMode_10(
            df["sales"].dropna().tolist()
        ),
        "DN_HistogramMode_5": catch22.DN_HistogramMode_5(df["sales"].dropna().tolist()),
        "DN_OutlierInclude_n_001_mdrmd": catch22.DN_OutlierInclude_n_001_mdrmd(
            df["sales"].dropna().tolist()
        ),
        "DN_OutlierInclude_p_001_mdrmd": catch22.DN_OutlierInclude_p_001_mdrmd(
            df["sales"].dropna().tolist()
        ),
        "FC_LocalSimple_mean1_tauresrat": catch22.FC_LocalSimple_mean1_tauresrat(
            df["sales"].dropna().tolist()
        ),
        "FC_LocalSimple_mean3_stderr": catch22.FC_LocalSimple_mean3_stderr(
            df["sales"].dropna().tolist()
        ),
        "IN_AutoMutualInfoStats_40_gaussian_fmmi": catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(
            df["sales"].dropna().tolist()
        ),
        "MD_hrv_classic_pnn40": catch22.MD_hrv_classic_pnn40(
            df["sales"].dropna().tolist()
        ),
        "PD_PeriodicityWang_th0_01": catch22.PD_PeriodicityWang_th0_01(
            df["sales"].dropna().tolist()
        ),
        "SB_BinaryStats_diff_longstretch0": catch22.SB_BinaryStats_diff_longstretch0(
            df["sales"].dropna().tolist()
        ),
        "SB_BinaryStats_mean_longstretch1": catch22.SB_BinaryStats_mean_longstretch1(
            df["sales"].dropna().tolist()
        ),
        "SB_MotifThree_quantile_hh": catch22.SB_MotifThree_quantile_hh(
            df["sales"].dropna().tolist()
        ),
        "SB_TransitionMatrix_3ac_sumdiagcov": catch22.SB_TransitionMatrix_3ac_sumdiagcov(
            df["sales"].dropna().tolist()
        ),
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(
            df["sales"].dropna().tolist()
        ),
        "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(
            df["sales"].dropna().tolist()
        ),
        "SP_Summaries_welch_rect_area_5_1": catch22.SP_Summaries_welch_rect_area_5_1(
            df["sales"].dropna().tolist()
        ),
        "SP_Summaries_welch_rect_centroid": catch22.SP_Summaries_welch_rect_centroid(
            df["sales"].dropna().tolist()
        ),
    }
).reset_index()


# %%
from kaggle_m5_forecasting.utils import timer

train_df = data[(data.d > 300) & (data.d < 1914)]
group_key = ["store_id", "item_id"]
with timer("mean"):
    df = train_df.groupby(group_key)["sales"].agg(
        {
            "fe_te_CO_Embed2_Dist_tau_d_expfit_meandiff": lambda x: catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(
                x.tolist()
            ),
            "fe_te_CO_f1ecac": lambda x: catch22.CO_f1ecac(x.tolist()),
            "fe_te_CO_FirstMin_ac": lambda x: catch22.CO_FirstMin_ac(x.tolist()),
            "fe_te_CO_HistogramAMI_even_2_5": lambda x: catch22.CO_HistogramAMI_even_2_5(
                x.tolist()
            ),
            "fe_te_CO_trev_1_num": lambda x: catch22.CO_trev_1_num(x.tolist()),
            "fe_te_DN_HistogramMode_10": lambda x: catch22.DN_HistogramMode_10(
                x.tolist()
            ),
            "fe_te_DN_HistogramMode_5": lambda x: catch22.DN_HistogramMode_5(
                x.tolist()
            ),
            "fe_te_DN_OutlierInclude_n_001_mdrmd": lambda x: catch22.DN_OutlierInclude_n_001_mdrmd(
                x.tolist()
            ),
            "fe_te_DN_OutlierInclude_p_001_mdrmd": lambda x: catch22.DN_OutlierInclude_p_001_mdrmd(
                x.tolist()
            ),
            "fe_te_FC_LocalSimple_mean1_tauresrat": lambda x: catch22.FC_LocalSimple_mean1_tauresrat(
                x.tolist()
            ),
            "fe_te_FC_LocalSimple_mean3_stderr": lambda x: catch22.FC_LocalSimple_mean3_stderr(
                x.tolist()
            ),
            "fe_te_IN_AutoMutualInfoStats_40_gaussian_fmmi": lambda x: catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(
                x.tolist()
            ),
            "fe_te_MD_hrv_classic_pnn40": lambda x: catch22.MD_hrv_classic_pnn40(
                x.tolist()
            ),
            "fe_te_PD_PeriodicityWang_th0_01": lambda x: catch22.PD_PeriodicityWang_th0_01(
                x.tolist()
            ),
            "fe_te_SB_BinaryStats_diff_longstretch0": lambda x: catch22.SB_BinaryStats_diff_longstretch0(
                x.tolist()
            ),
            "fe_te_SB_BinaryStats_mean_longstretch1": lambda x: catch22.SB_BinaryStats_mean_longstretch1(
                x.tolist()
            ),
            "fe_te_SB_MotifThree_quantile_hh": lambda x: catch22.SB_MotifThree_quantile_hh(
                x.tolist()
            ),
            "fe_te_SB_TransitionMatrix_3ac_sumdiagcov": lambda x: catch22.SB_TransitionMatrix_3ac_sumdiagcov(
                x.tolist()
            ),
            "fe_te_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": lambda x: catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(
                x.tolist()
            ),
            "fe_te_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": lambda x: catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(
                x.tolist()
            ),
            "fe_te_SP_Summaries_welch_rect_area_5_1": lambda x: catch22.SP_Summaries_welch_rect_area_5_1(
                x.tolist()
            ),
            "fe_te_SP_Summaries_welch_rect_centroid": lambda x: catch22.SP_Summaries_welch_rect_centroid(
                x.tolist()
            ),
        }
    )

df

# %%
d = data[(data.d > 300) & (data.d < 500)]["sales"].values
