import calendar
import logging
from typing import Union

import pandas as pd

from src.utils import create_comma_separated_string


def get_db_params(data_resolution: str, months: Union[int, list[int]]) -> dict:
    """
    Generates a configuration dictionary for database access based on the selected data resolution and a month selection.

    This function is used to dynamically generate database parameters for further processing (e.g., results and
    ranking tables), taking into account seasonal or user-defined periods.

    The following dictionary entries are generated:
     - 'table_name_results': Table name for results, formed from data_resolution (e.g., 'hourly_results').
     - 'table_name_ranking': Table name for rankings, also based on data_resolution.
     - 'period': Description of the period under consideration, depending on the months parameter:
        - Month name for single month
        - 'Year' for all 12 months
        - Season name ('Winter', 'Spring', ‘Summer’, 'Fall') for matching month combination
        - Otherwise 'undefined'
     - 'months': CSV with monthly figures

    :param data_resolution: Resolution of the data, e.g., “hourly” or “daily.”
    :param months: Single month (e.g., 1) or list of several months (e.g., [6, 7, 8]).
    :return: A dictionary with the following entries:
    """
    saison_dict = {
        tuple([1, 2, 12]): "Winter",
        tuple([3, 4, 5]): "Spring",
        tuple([6, 7, 8]): "Summer",
        tuple([9, 10, 11]): "Fall"
    }
    params = {"table_name_results": f"{data_resolution}_results",
              "table_name_ranking": f"{data_resolution}_rank_forward_selection",
              "table_name_model": f"{data_resolution}_models"}

    # Define period
    if type(months) == int:
        params["period"] = calendar.month_name[months]
    elif sorted(months) == sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
        params["period"] = "Year"
    else:
        if saison_dict.get(tuple(sorted(months))):
            params["period"] = f"{saison_dict[tuple(sorted(months))]}"
        else:
            params["period"] = "undefined"

    # Save monthly figures additionally as csv
    if type(months) == int:
        params["months"] = str(months)
    else:
        params["months"] = ';'.join(map(str, months))

    return params


def convert_timestamp(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Converts timestamp columns of a DataFrame into real datetime objects, depending on the
    specified mode (‘daily’ or ‘hourly’).

    If an unsupported mode is used, the program will terminate with a critical log entry.
    :param df: The input DataFrame, which must contain a column ‘date’ (for daily) or ‘datetime’ (for hourly).
    :param mode: ‘daily’ = ‘%Y-%m-%d’ or 'hourly' = '%Y-%m-%d %H:%M:%S'
    :return: DataFrame with the converted time column.
    """
    # Conversion to real timestamp
    if mode == 'daily':
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    elif mode == 'hourly':
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    else:
        logging.critical('Unsupported mode for timestamp conversion.')
        exit(1)

    return df


def cast_string_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Convert specified  columns of a DataFrame to float (if necessary) and remove missing values.

    Some values appear to be stored as strings in the database when the weather tables are updated
    via the Open Meteo API.
    :param df: The input DataFrame.
    :param features: List of features (columns) to convert.
    :return: DataFrame with the converted values.
    """
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    return df


def _logging_months(months: Union[int, list[int]]):
    """
    Logs the selected month(s) as readable names using 'calendar.month_name':
     - Single int -> single month name
     - List of ints -> comma-separated month names via 'create_comma_separated_string(...)'
    :param months: Months to which the training data is limited
    """
    if type(months) == int:
        logging.info(f'Month: {calendar.month_name[months]}')
    else:
        months_names = [calendar.month_name[month] for month in months]
        logging.info(f'Months: {create_comma_separated_string(months_names)}')
