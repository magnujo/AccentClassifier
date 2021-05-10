from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def get_label_from_frame(df: DataFrame, column: str):
    distinct = df[column].unique()

def clean_binarize(dataframe, focus_col, values_to_remove=["other"]):
    df = dataframe[["path", focus_col]]
    df = df.dropna()
    for val in values_to_remove: #ved ikke om der er en smartere løsning
        df = df[df[focus_col] != val]
    lb = LabelBinarizer()
    label_binarizer = lb.fit_transform(df[focus_col])
    return df, label_binarizer

def get_info(dataframe, focus_col):
    df = dataframe[["path", focus_col]]
    df = df.dropna()
    print(df.shape[1])
    print(df[focus_col].unique())
    print(df[focus_col].value_counts())
    sns.countplot(y=focus_col, data=df)
    plt.show()