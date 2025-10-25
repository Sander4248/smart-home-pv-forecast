from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Set

import pandas as pd

from src.config.paths import WEATHER_FORECAST_DB_PATH
from src.utils import get_cursor


# ----------------------------
# Getter
# ----------------------------

def get_all_response_timestamps() -> Set[datetime]:
    """
    Get all actual response timestamps from the database.

    :return: Set with all actual response timestamps.
    """
    with get_cursor(WEATHER_FORECAST_DB_PATH) as cursor:
        query = """
            SELECT DISTINCT response_date, response_hour
              FROM forecast
             ORDER BY response_date, response_hour
        """
        cursor.execute(query)
        rows = cursor.fetchall()

    # Parse into datetime objects and pack into a set
    available_response_timestamps = {
        datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
        for date, time in rows
    }

    return available_response_timestamps


def get_min_forecast_ts() -> datetime:
    """
    Determine the minimum forecast timestamp.

    :return: Minimum forecast timestamp.
    """
    query = """
        SELECT MIN(forecast_hour), forecast_date
          FROM forecast
         WHERE forecast_date = (
               SELECT MIN(forecast_date)
                 FROM forecast
               )
        """
    with get_cursor(WEATHER_FORECAST_DB_PATH) as cursor:
        cursor.execute(query)
        min_hour, min_date = cursor.fetchone()
    min_forecast_ts = datetime.strptime(f"{min_date} {min_hour}", "%Y-%m-%d %H:%M:%S")

    return min_forecast_ts


def get_forecast_weather_data_by_timestamp(forecast_date: str,
                               forecast_hour: str,
                               response_date: str,
                               response_hour: str
                               ) -> pd.DataFrame:
    """
    Retrieves weather forecast data for a specific date and hour from the database.

    :param forecast_date: Date of the weather forecast, format “%Y-%m-%d”
    :param forecast_hour: Time of the weather forecast, format “%H:%M:%S”
    :param response_date: Date of the response, format “%Y-%m-%d”
    :param response_hour: Time of the response, format “%H:%M:%S”
    :return: pandas.DataFrame with the weather forecast. If no data matches, an empty DataFrame is returned.
    """
    query = f"""
                SELECT *
                  FROM forecast
                 WHERE response_date = '{response_date}'
                   AND response_hour = '{response_hour}'
                   AND forecast_date = '{forecast_date}'
                   AND forecast_hour = '{forecast_hour}'
            """
    with sqlite3.connect(WEATHER_FORECAST_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df

# ----------------------------
# Setter
# ----------------------------
