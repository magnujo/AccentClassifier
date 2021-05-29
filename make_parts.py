import os
from pathlib import Path

from pandas import DataFrame
import pandas as pd
from shutil import copyfile


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_folder(path1: str, path2: str, df: DataFrame):
    os.chdir(path1)

    files = os.listdir()
    count = 0
    n = len(files)
    parts = split(files, 100)
    # You dont need the number of files in the folder, just iterate over them directly using:
    i_part = 1
    for part in parts:
        part_name = f"part{i_part}"
        print(part_name)
        if not os.path.exists(Path(path2, part_name)):
            os.makedirs(Path(path2, part_name))

        for file in part:
            name, ext = os.path.splitext(file)
            df.loc[df['path'] == name + ".mp3", 'part'] = part_name
            copyfile(file, Path(Path(path2, part_name), name + ext))
        i_part += 1

    print("done")


if __name__ == "__main__":
    df = pd.read_csv(r"D:\data_small\df_accent.csv")
    split_folder(r"D:\data_small\cv-corpus-6.1-2020-12-11\en\wav",
                 r"D:\data_small\cv-corpus-6.1-2020-12-11\en", df)
    df.to_csv(r"D:\data_small\df_accent.csv")
