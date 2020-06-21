import sklearn.decomposition
import pandas as pd
from kaggle_m5_forecasting import M5
from kaggle_m5_forecasting.data.make_data import MakeData
from kaggle_m5_forecasting.data.fe_catch22 import FECatch22
import sklearn.preprocessing


class FECatch22PCA(M5):
    def requires(self):
        return dict(data=MakeData(), catch22_df=FECatch22())

    def run(self):

        data: pd.DataFrame = self.load("data")
        catch22_df: pd.DataFrame = self.load("catch22_df")

        catch22_df = catch22_df.drop(["percentile75", "percentile100", "std"], axis=1)
        catch22_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(
            catch22_df.values
        )

        n_components = 3
        catch22_df = pd.concat(
            [
                catch22_df.reset_index(),
                pd.DataFrame(
                    sklearn.decomposition.PCA(n_components=n_components).fit_transform(
                        catch22_scaled
                    ),
                    columns=[f"fe_catch22_pca_{i}" for i in range(n_components)],
                ),
            ],
            axis=1,
        )
        df = data.merge(
            catch22_df[["id"] + [f"fe_catch22_pca_{i}" for i in range(n_components)]],
            on="id",
            how="left",
        ).filter(like="fe_catch22_pca")
        print(df.info())
        self.dump(df)
