from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

from read_file import extract, convert_mp3s_to_wavs

seed = 42


def find_class_with_maximum_rows(df: DataFrame, label_col: str, allowed_labels: list):
    df_lab = df[df[label_col].isin(allowed_labels)]
    group = df_lab.groupby([label_col])
    size = group.size()
    mean = 50000
    for lab, _ in group:
        count = size[lab]
        diff = int(np.ceil(mean - count))
        if diff > 0:
            df_new = select_random_row(df, label_col, lab, diff)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            diff = np.abs(diff)
            df_lab = df[df[label_col] == lab]
            drop_indicies = np.random.choice(df_lab.index, diff, replace=False)
            df = df.drop(drop_indicies)

    df = df[df[label_col].isin(allowed_labels)]
    return df


def select_random_row(df: DataFrame, label: str, val: str, n_new_rows: int):
    df = df[df[label] == val]
    df_new = df.sample(n_new_rows, replace=True, random_state=seed)
    return df_new


def down_up_sample(path_to_csv: str, save_dir: str, label_col: str, allowed_labels: List[str]):
    np.random.seed(seed)
    df = pd.read_csv(path_to_csv)
    sns.countplot(y=label_col, data=df)
    plt.savefig(Path(save_dir, "before.png"))
    plt.show()
    res = find_class_with_maximum_rows(df, label_col, allowed_labels)
    res.to_csv(Path(save_dir, f"df_{label_col}.png"))
    sns.countplot(y=label_col, data=res)
    plt.savefig(Path(save_dir, "after.png"))
    plt.show()
    return res


if __name__ == "__main__":
    res = down_up_sample(r"D:\data\cv-corpus-6.1-2020-12-11\en\df_accent.csv", r"I:\accent_300K", "accent",
                   ["us", "indian", "england", "canada", "australia"])
    res.to_csv(r"I:\accent_300K\df_accent.csv")
    paths = res["path"].tolist()
    paths.sort()
    extract(paths, r"I:\accent_300K", r"C:\Users\hadis\Downloads\en.tar")
    convert_mp3s_to_wavs(r"I:\accent_300K\cv-corpus-6.1-2020-12-11\en\clips")

