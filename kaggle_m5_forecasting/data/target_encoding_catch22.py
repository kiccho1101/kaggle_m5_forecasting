from kaggle_m5_forecasting.base import M5
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.config import Config
from kaggle_m5_forecasting.utils import timer
import pandas as pd
import catch22
from tqdm.autonotebook import tqdm
from typing import List, Tuple


def target_encoding_catch22(train_df: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        ["store_id", "item_id"],
    ]
    result: List[Tuple[List[str], pd.DataFrame]] = []
    with timer("target encoding"):
        for group_key in tqdm(group_keys):
            with timer("{} te".format(str(group_key))):
                tmp_df = train_df.groupby(group_key)["sales"].agg(
                    {
                        "fe_te_CO_Embed2_Dist_tau_d_expfit_meandiff": lambda x: catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(
                            x.tolist()
                        ),
                        "fe_te_CO_f1ecac": lambda x: catch22.CO_f1ecac(x.tolist()),
                        "fe_te_CO_FirstMin_ac": lambda x: catch22.CO_FirstMin_ac(
                            x.tolist()
                        ),
                        "fe_te_CO_HistogramAMI_even_2_5": lambda x: catch22.CO_HistogramAMI_even_2_5(
                            x.tolist()
                        ),
                        "fe_te_CO_trev_1_num": lambda x: catch22.CO_trev_1_num(
                            x.tolist()
                        ),
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
                tmp_df.reset_index(inplace=True, drop=True)
                result.append((group_key, tmp_df))
    return result


class TECatch22ValData(M5):
    def requires(self):
        return dict(data=MakeData())

    def run(self):
        config = Config()
        data: pd.DataFrame = self.load("data")
        results: List[List[Tuple[List[str], pd.DataFrame]]] = []
        for end_day in config.CV_START_DAYS:
            train_df: pd.DataFrame = data[
                (data.d > config.START_DAY) & (data.d < end_day)
            ]
            results.append(target_encoding_catch22(train_df))
        self.dump(results)


class TECatch22Data(M5):
    def requires(self):
        return MakeData()

    def run(self):
        config = Config()
        data: pd.DataFrame = self.load()
        train_df: pd.DataFrame = data[(data.d > config.START_DAY) & (data.d <= 1913)]
        result: List[Tuple[List[str], pd.DataFrame]] = target_encoding_catch22(train_df)
        self.dump(result)
