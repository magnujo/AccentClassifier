from pandas import DataFrame


def get_label_from_frame(df: DataFrame, column: str):
    distinct = df[column].unique()