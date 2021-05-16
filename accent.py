# %%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from read_file import extract

df = pd.read_csv(Path().joinpath("data", "validated.tsv"), sep="\t")


def getAccentDF(dataframe):
    return dataframe[["path", "accent"]]


# %%
df_accent = getAccentDF(df)

# %%
df_accent.dropna(inplace=True)
df_accent = df_accent[df_accent.accent != "other"]

# %%

sns.countplot(y="accent", data=df_accent)
#plt.show()
df_accent.accent.value_counts()

# %%
df_accent.shape

# %%
# delete values inrow with very low frequency:
df_accent = df_accent[df_accent.accent != "southatlandtic"]
df_accent = df_accent[df_accent.accent != "bermuda"]
df_accent = df_accent[df_accent.accent != "malaysia"]
df_accent = df_accent[df_accent.accent != "hongkong"]
df_accent = df_accent[df_accent.accent != "philippines"]
df_accent = df_accent[df_accent.accent != "ireland"]
df_accent = df_accent[df_accent.accent != "wales"]
df_accent = df_accent[df_accent.accent != "singapore"]


df_accent.describe()
print(df_accent.head(2))
print(df_accent.accent.value_counts())
# %%
distinct_accents = df_accent["accent"].unique()
print(distinct_accents)

lb = LabelBinarizer()

label_binarizer = lb.fit_transform(df_accent["accent"])

print(label_binarizer)

# df_accent.to_csv(r"D:\data\cv-corpus-6.1-2020-12-11\en\df_accent.csv")

sample = df_accent.sample(n=100)
sample.to_csv(r"D:\data_small\df_accent_small.csv")

extract(sample["path"].tolist())

X = df_accent[:, df_accent!='']