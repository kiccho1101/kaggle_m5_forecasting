# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from thunderbolt import Thunderbolt

tb = Thunderbolt("./../../resource")
data: pd.DataFrame = tb.get_data("MakeData")

# %%

df = data[data.id == data.loc[np.random.randint(0, len(data) - 1), "id"]].reset_index(
    drop=True
)

lag = 7
w_sizes = [7, 30, 60, 90, 180]
w_sizes = [30]
alpha = 0.03
for w_size in w_sizes:
    df[f"r_t{lag}_{w_size}"] = df["sales"].shift(lag).rolling(w_size).mean()
    df[f"r_ewm{alpha}_t{lag}_{w_size}"] = df["sales"].shift(lag).ewm(alpha=alpha).mean()
features = (
    ["sales"]
    + [f"r_t{lag}_{w_size}" for w_size in w_sizes]
    + [f"r_ewm{alpha}_t{lag}_{w_size}" for w_size in w_sizes]
)
df[features].tail(200).plot()
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
).sort_values("rmse")


# %%
test = data[(data.d > 1885) & (data.d < 1914)]


# %%
import lightgbm as lgb
