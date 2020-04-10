# %%
from thunderbolt import Thunderbolt

tb = Thunderbolt("./resource")
data = tb.get_data("FERollingSum")


# %%
list(data)


# %%
import pandas as pd

final = pd.read_csv("./output/submission/submission_2020-04-10_lgbm_3.csv")

# %%

for i in range(1, 29):
    final["F" + str(i)] *= 1.02 / 1.04
final.to_csv("./output/submission/submission_2020-04-10_lgbm_6.csv", index=False)


# %%
