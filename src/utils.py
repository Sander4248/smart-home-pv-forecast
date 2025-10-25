import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.config.paths import WEATHER_FORECAST_DB_PATH


@contextmanager
def get_cursor(db: str):
    """
    Context manager to open a SQLite connection with foreign key support.
    Ensures automatic commit and close of the connection.
    Enables foreign key checking.
    :param db: Database path
    :return:
    """
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    finally:
        conn.close()


def append_time_features(df: pd.DataFrame, column_name: str = "datetime") -> pd.DataFrame:
    """
    Adds cyclical time-based features from a datetime to a dataframe.
    :param df: DataFrame containing a datetime column with hourly timestamps.
    :param column_name: Name of column with timestamp
    :return: DataFrame with added cyclic timestamp features.
    """
    df["hour"] = df[column_name].dt.hour + df[column_name].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def append_date_features(df: pd.DataFrame, column_name: str = "date") -> pd.DataFrame:
    """
    Adds cyclical date features to a dataframe.
    :param df: DataFrame containing a date column with daily timestamps.
    :param column_name: Name of column with datestamp
    :return: DataFrame with added cyclic day-of-year features.
    """
    df["day_of_year"] = df[column_name].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def create_comma_separated_string(values: list | set) -> str:
    """
    Combine lists into comma-separated strings e.g. for SQL statement
    :param values: List with values to combine
    :return: Combined string
    """
    return ', '.join(map(str, values))
