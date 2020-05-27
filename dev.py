# %%
import pickle
import pandas as pd

df_0: pd.DataFrame = pickle.load(open("cv_pred_0.pkl", "rb"))
df_reg_0: pd.DataFrame = pickle.load(open("cv_pred_reg_0.pkl", "rb"))


# %%
import sklearn.metrics

threshold = 0.9
print((df_0["sales"] == 0).sum())
print((df_0["sales"] != 1).sum())
print((df_0["sales_is_zero_proba"] >= threshold).sum())
y_true = (df_0["sales"] == 0).astype(int).values
y_pred = (df_0["sales_is_zero_proba"] >= threshold).astype(int).values
print(sklearn.metrics.precision_score(y_true, y_pred))
print(sklearn.metrics.recall_score(y_true, y_pred))

# %%
test_pred = pickle.load(open("./test_pred.pkl", "rb"))


# %%


# %%
r = 0.03
n = 10
y = 30
s = 0
for i in range(y):
    s += n * 12
    s *= 1.03


# %%
s
