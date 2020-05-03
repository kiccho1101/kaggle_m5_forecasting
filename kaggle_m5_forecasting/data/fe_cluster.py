from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.utils import timer
from kaggle_m5_forecasting.data.make_data import MakeData
import pandas as pd
import numpy as np
import sklearn.cluster
import sklearn.preprocessing


class FECluster(M5):
    def requires(self):
        return MakeData()

    def run(self):
        data: pd.DataFrame = self.load()

        with timer("calc aggregates groupby id"):
            methods = ["mean", "std", "min", "max", "median"]
            group = data.groupby(["id"])
            df = group["sales"].agg(methods)
            df["zero_count"] = group["sales"].apply(lambda x: np.count_nonzero(x == 0))
            df["nonzero_count"] = group["sales"].apply(lambda x: np.count_nonzero(x))

        with timer("calc aggregates groupby id and dw"):
            group = data.groupby(["tm_dw", "id"])
            group_df = group["sales"].agg(methods)
            for dw in range(0, 7):
                df = pd.concat(
                    [
                        df,
                        group_df.loc[dw].rename(
                            {col: f"tm_dw_{dw}_{col}" for col in group_df.columns},
                            axis=1,
                        ),
                    ],
                    axis=1,
                )

        with timer("k-means clustering"):
            scaler = sklearn.preprocessing.StandardScaler()
            cl = sklearn.cluster.KMeans(n_clusters=12)
            cluster = cl.fit_predict(scaler.fit_transform(df))
            df["fe_cluster"] = cluster
            df = df["fe_cluster"].reset_index()

        with timer("merge data"):
            data = data.merge(df, on=["id"], how="left")
            data["fe_cluster"] = data["fe_cluster"].astype(np.int8)

        df = data.filter(like="fe_cluster")
        print(df.info())
        self.dump(df)
