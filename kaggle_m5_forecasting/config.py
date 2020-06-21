# %%
from typing import List, Dict, Any


class Config:

    SEED = 402
    MAX_LAGS = 100
    START_DAY = 300
    num_boost_round = 250
    MIN_SUM = 0
    CV_START_DAYS = [1914, 1886, 1858]
    CV_SAMPLE_RATE = 0.5
    DROP_NA = False
    DROP_OUTLIERS = True
    TARGET = "sales"
    CLS_THRESHOLD = 0.9

    lgbm_params: Dict[str, Any] = {
        "boosting_type": "gbdt",
        "metric": "rmse,mae,tweedie",
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "boost_from_average": False,
        "n_jobs": -1,
        "seed": SEED,
        # "max_depth": 80,
        "num_leaves": 157,
        "min_data_in_leaf": 301,
        "learning_rate": 0.03,
        "bagging_freq": 1,
        # "max_bin": 100,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.5,
        # "feature_fraction": 0.4,
        # "bagging_fraction": 0.45,
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
        "snap_CA",
        "snap_TX",
        "snap_WI",
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
        "fe_price_change_t365",
        "fe_price_discount",
        "fe_price_discount_rate",
        "fe_price_skew",
        "fe_price_momentum",
        "fe_price_momentum_m",
        "fe_price_momentum_y",
        "shift_t7",
        "shift_t28",
        "shift_t29",
        "shift_t30",
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
        "fe_rolling_skew_t28_30",
        "fe_rolling_kurt_t28_30",
        "fe_te_store_id_item_id_mean",
        "fe_te_store_id_item_id_std",
        "fe_te_store_id_cat_id_mean",
        "fe_te_store_id_cat_id_std",
        "fe_te_store_id_dept_id_mean",
        "fe_te_store_id_dept_id_std",
        "fe_te_store_id_item_id_tm_dw_mean",
        "fe_te_store_id_item_id_tm_dw_std",
        "fe_te_item_id_tm_dw_mean",
        "fe_te_item_id_tm_dw_std",
        "fe_catch22_pca_0",
        "fe_catch22_pca_1",
        "fe_catch22_pca_2",
        "fe_weather_mintempC",
        # "fe_weather_maxtempC",
        # "fe_weather_humidity",
        # "fe_weather_sunHour",
        # "fe_weather_cloudcover",
        "fe_unemployment",
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
