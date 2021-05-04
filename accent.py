# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from test import extract

df = pd.read_csv(Path().joinpath("data", "validated.tsv"), sep="\t")


def getAccentDF(dataframe):
    return dataframe[["path", "accent"]]


# %%
df_accent = getAccentDF(df)

# %%
df_accent.dropna(inplace=True)

# %%

sns.countplot(y="accent", data=df_accent)
plt.show()
df_accent.accent.value_counts()

# %%
df_accent.shape

# %%
df_accent.describe()

# %%
distinct_accents = df_accent["accent"].unique()
print(distinct_accents)

lb = LabelBinarizer()

label_binarizer = lb.fit_transform(df_accent["accent"])

print(label_binarizer)

