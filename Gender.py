from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from read_file import extract

df = pd.read_csv(Path().joinpath("data", "validated.tsv"), sep="\t")
print(df.shape)
df = df[["path", "gender"]]

df.dropna(inplace=True)
df = df[df.gender != "other"]
#df["gender"].value_counts()
df.describe()


sns.countplot(y="gender", data=df)
plt.show()
print(df.count())

df.to_csv(r"I:\Gender\df_gender.csv")
extract(df["path"].tolist())


