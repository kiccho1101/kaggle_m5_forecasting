from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.make_data import MakeData
import pandas as pd
import numpy as np
import scipy
import catch22
import sklearn.preprocessing


class FECatch22(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()
        data = data[data.d < 1942]

        with timer("calc grouped aggregates"):
            catch22_df = data.groupby(["id"])["sales"].agg(
                mean=lambda x: x.dropna().values.mean(),
                percentile25=lambda x: x.dropna()
                .sort_values()[: int(len(x) * 0.25)]
                .mean(),
                percentile50=lambda x: x.dropna()
                .sort_values()[int(len(x) * 0.25) : int(len(x) * 0.5)]
                .mean(),
                percentile75=lambda x: x.dropna()
                .sort_values()[int(len(x) * 0.5) : int(len(x) * 0.75)]
                .mean(),
                percentile100=lambda x: x.dropna()
                .sort_values()[int(len(x) * 0.75) :]
                .mean(),
                std=lambda x: x.dropna().values.std(),
                CO_Embed2_Dist_tau_d_expfit_meandiff=lambda x: catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(
                    x.dropna().tolist()
                ),
                CO_f1ecac=lambda x: catch22.CO_f1ecac(x.dropna().tolist()),
                CO_FirstMin_ac=lambda x: catch22.CO_FirstMin_ac(x.dropna().tolist()),
                CO_HistogramAMI_even_2_5=lambda x: catch22.CO_HistogramAMI_even_2_5(
                    x.dropna().tolist()
                ),
                CO_trev_1_num=lambda x: catch22.CO_trev_1_num(x.dropna().tolist()),
                DN_HistogramMode_10=lambda x: catch22.DN_HistogramMode_10(
                    x.dropna().tolist()
                ),
                DN_HistogramMode_5=lambda x: catch22.DN_HistogramMode_5(
                    x.dropna().tolist()
                ),
                DN_OutlierInclude_n_001_mdrmd=lambda x: catch22.DN_OutlierInclude_n_001_mdrmd(
                    x.dropna().tolist()
                ),
                DN_OutlierInclude_p_001_mdrmd=lambda x: catch22.DN_OutlierInclude_p_001_mdrmd(
                    x.dropna().tolist()
                ),
                FC_LocalSimple_mean1_tauresrat=lambda x: catch22.FC_LocalSimple_mean1_tauresrat(
                    x.dropna().tolist()
                ),
                FC_LocalSimple_mean3_stderr=lambda x: catch22.FC_LocalSimple_mean3_stderr(
                    x.dropna().tolist()
                ),
                IN_AutoMutualInfoStats_40_gaussian_fmmi=lambda x: catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(
                    x.dropna().tolist()
                ),
                MD_hrv_classic_pnn40=lambda x: catch22.MD_hrv_classic_pnn40(
                    x.dropna().tolist()
                ),
                PD_PeriodicityWang_th0_01=lambda x: catch22.PD_PeriodicityWang_th0_01(
                    x.dropna().tolist()
                ),
                SB_BinaryStats_diff_longstretch0=lambda x: catch22.SB_BinaryStats_diff_longstretch0(
                    x.dropna().tolist()
                ),
                SB_BinaryStats_mean_longstretch1=lambda x: catch22.SB_BinaryStats_mean_longstretch1(
                    x.dropna().tolist()
                ),
                SB_MotifThree_quantile_hh=lambda x: catch22.SB_MotifThree_quantile_hh(
                    x.dropna().tolist()
                ),
                SB_TransitionMatrix_3ac_sumdiagcov=lambda x: catch22.SB_TransitionMatrix_3ac_sumdiagcov(
                    x.dropna().tolist()
                ),
                SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1=lambda x: catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(
                    x.dropna().tolist()
                ),
                SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1=lambda x: catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(
                    x.dropna().tolist()
                ),
                SP_Summaries_welch_rect_area_5_1=lambda x: catch22.SP_Summaries_welch_rect_area_5_1(
                    x.dropna().tolist()
                ),
                SP_Summaries_welch_rect_centroid=lambda x: catch22.SP_Summaries_welch_rect_centroid(
                    x.dropna().tolist()
                ),
            )

        print(catch22_df.info())
        self.dump(catch22_df)
