# %%
from pathlib import Path

import pandas as pd
import seaborn as sns

df = pd.read_csv(Path().joinpath("data", "validated.tsv"), sep="\t")


def getAccentDF(dataframe):
    return dataframe[['client_id', "path", "accent"]]


# %%
df_accent = getAccentDF(df)

# %%
df_accent.dropna(inplace=True)

# %%
from matplotlib import pyplot as plt

sns.countplot(y="accent", data=df_accent)
plt.show()
df_accent.accent.value_counts()

# %%
df_accent.shape

# %%
df_accent.describe()

# %%
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df_accent[["accent"]])

# %%
ohe.categories_
ohe = OneHotEncoder(sparse=False).fit_transform(df_accent[["accent"]])


# %%

print(ohe.categories_)
