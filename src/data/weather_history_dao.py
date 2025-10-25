from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from src.config.paths import WEATHER_HISTORY_DB_PATH
from src.config.schema import ProjectParams


# ----------------------------
# Getter
# ----------------------------

def get_historical_weather_data_in_range(months: str,
                                         features: str,
                                         limit_training_data: bool
                                         ) -> pd.DataFrame:
    """
    Loads the hourly weather data for selected months from the database

    If all data is to be loaded, all months (as numbers) must be entered in the ‘months_str’ parameter.
    All data for the filtered months is output regardless of the year.

    If limit_training_data = True, no historical data will be loaded from the period for which forecast data exists.

    :param months: Comma separated string of months e.g. '1, 2, 3'
    :param features: Comma separated string of features (columns) to be loaded
    :param limit_training_data: Limits the training data to the period for which no weather forecast data is available
    :return: DataFrame with hourly weather data
    """
    cfg = ProjectParams()

    if limit_training_data:
        max_date = cfg.START_FORECAST_DATE
    else:
        max_date = datetime.today().strftime("%Y-%m-%d")

    query = f"""
        select datetime(date || ' ' || hour) as datetime, {features}
          from history
         where strftime('%m', date) + 0  in ({months})
           and date < '{max_date}'
    """

    with sqlite3.connect(WEATHER_HISTORY_DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def get_historical_weather_data_by_timestamp(forecast_date: str,
                                             forecast_hour: str
                                             ) -> pd.DataFrame:
    """
    Retrieves weather history data for a specific date and hour from the database.

    :param forecast_date: Date of the weather history, format “%Y-%m-%d”
    :param forecast_hour: Time of the weather history, format “%H:%M:%S”
    :return: pandas.DataFrame with the weather history. If no data matches, an empty DataFrame is returned.
    """
    query = f"""
                SELECT *
                  FROM history
                 WHERE date = '{forecast_date}'
                   AND hour = '{forecast_hour}'
            """
    with sqlite3.connect(WEATHER_HISTORY_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)
    return df

# ----------------------------
# Setter
# ----------------------------
