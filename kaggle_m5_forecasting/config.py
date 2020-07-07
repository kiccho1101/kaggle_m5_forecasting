# %%
from typing import List, Dict, Any


class Config:

    SEED = 700
    MAX_LAGS = 100
    START_DAY = 0
    num_boost_round = 1000
    MIN_SUM = 0
    # CV_START_DAYS = [1914, 1886, 1858]
    CV_START_DAYS = [1942]
    CV_SAMPLE_RATE = 0.5
    DROP_NA = False
    DROP_OUTLIERS = True
    TARGET = "sales"
    CLS_THRESHOLD = 0.7
    CLS_TIMESTAMP = "2020-06-26_21:38:59"
    CLS_POSTPROCESSING = False
    MODEL = "store"

    lgbm_params: Dict[str, Any] = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "boost_from_average": False,
        "n_jobs": -1,
        "seed": SEED,
        # "max_depth": 80,
        "num_leaves": 157,
        "min_data_in_leaf": 300,
        "learning_rate": 0.03,
        "bagging_freq": 1,
        # "max_bin": 100,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.5,
    }

    lgbm_cls_params: Dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "n_jobs": -1,
        "seed": SEED,
        "num_leaves": 2500,
        "min_data_in_leaf": 1000,
        "learning_rate": 0.03,
        "bagging_freq": 1,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.5,
        "n_estimators": num_boost_round,
    }

    features = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        # "fe_event",
        "fe_event_strength",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "snap",
        "tm_d",
        "tm_w",
        "tm_m",
        "tm_y",
        "tm_quarter",
        "tm_wm",
        "tm_dw",
        "tm_w_end",
        "sell_price",
        "fe_rolling_price_std_t7",
        "fe_rolling_price_std_t30",
        "fe_price_change_t1",
        # "fe_price_change_t365",
        "fe_price_discount",
        "fe_price_discount_rate",
        "fe_price_skew",
        "fe_price_momentum",
        "fe_price_momentum_m",
        "fe_price_momentum_y",
        "shift_t7",
        "shift_t8",
        "shift_t28",
        "shift_t29",
        "shift_t30",
        "shift_t31",
        "shift_t32",
        "shift_t33",
        "shift_t34",
        "shift_t35",
        "shift_t36",
        "shift_t37",
        "shift_t38",
        "shift_t39",
        "shift_t40",
        "shift_t41",
        "shift_t42",
        # "fe_rolling_mean_t1_7",
        # "fe_rolling_mean_t1_30",
        # "fe_rolling_mean_t1_60",
        "fe_rolling_mean_t7_7",
        "fe_rolling_mean_t7_30",
        "fe_rolling_mean_t7_60",
        "fe_rolling_mean_t28_7",
        "fe_rolling_mean_t28_30",
        "fe_rolling_mean_t28_60",
        "fe_rolling_mean_t28_90",
        "fe_rolling_mean_t28_180",
        "fe_rolling_std_t28_7",
        "fe_rolling_std_t28_30",
        "fe_rolling_std_t28_60",
        "fe_rolling_std_t28_90",
        # "fe_rolling_dw_mean_t4_4",
        # "fe_rolling_dw_mean_t4_8",
        "fe_te_item_id_mean",
        "fe_te_item_id_std",
        "fe_te_item_id_tm_dw_mean",
        "fe_te_item_id_tm_dw_std",
        "fe_te_state_id_item_id_mean",
        "fe_te_state_id_item_id_std",
        "fe_te_store_id_item_id_mean",
        "fe_te_store_id_item_id_std",
        "fe_te_store_id_item_id_snap_mean",
        "fe_te_store_id_item_id_snap_std",
        "fe_te_store_id_item_id_snap_tm_dw_mean",
        "fe_te_store_id_item_id_snap_tm_dw_std",
        # "fe_te_store_id_item_id_fe_event_dw_mean",
        "fe_te_state_id_item_id_snap_tm_dw_mean",
        "fe_te_state_id_item_id_snap_tm_dw_std",
        "fe_te_store_id_item_id_tm_dw_mean",
        "fe_te_store_id_item_id_tm_dw_std",
        "fe_catch22_pca_0",
        "fe_catch22_pca_1",
        "fe_catch22_pca_2",
        "fe_rolling_store_id_mean_28_30",
        "fe_rolling_store_id_std_28_30",
    ]

    lgbm_cat_features: List[str] = [
        f
        for f in features
        if f
        in [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "fe_event",
            "event_name_1_yesterday",
            "event_type_1_yesterday",
            "event_name_1_tomorrow",
            "event_type_1_tomorrow",
            "snap",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    ]
