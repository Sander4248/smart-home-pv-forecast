from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

import pandas as pd

from src.config.paths import SOLAR_POWER_DB_PATH
from src.utils import get_cursor


# ----------------------------
# Getter
# ----------------------------

def get_hourly_yield_in_range(months: str) -> pd.DataFrame:
    """
    Loads the hourly PV data for selected months from the database

    If all data is to be loaded, all months (as numbers) must be entered in the ‘months_str’ parameter.
    All data for the filtered months is output regardless of the year.
    :param months: Comma separated string of months e.g. '1, 2, 3'
    :return: DataFrame with hourly PV data
    """
    query = f"""
        select datetime(date || ' ' || time) as datetime, pv_yield_calc as pv_yield
          from hourly_calculated
         where strftime('%m', date) + 0  in ({months})
    """

    with sqlite3.connect(SOLAR_POWER_DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def get_daily_yield_in_range(months: str) -> pd.DataFrame:
    """
    Loads the daily PV data for selected months from the database

    If all data is to be loaded, all months (as numbers) must be entered in the ‘months_str’ parameter.
    All data for the filtered months is output regardless of the year.
    :param months: Comma separated string of months e.g. '1, 2, 3'
    :return: DataFrame with daily PV data
    """
    query = f"""
        select date(date) as date, pv_yield
          from monthly_reports
         where strftime('%m', date) + 0  in ({months})
    """

    with sqlite3.connect(SOLAR_POWER_DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def get_max_solar_timestamp(data_resolution: str) -> datetime:
    """
    Determine the maximum timestamp to narrow down the determination of the evaluation timestamps.

    :param data_resolution: Resolution of the dataset. Must be either "hourly" or "daily"
    :return: Latest PV data date
    """
    if data_resolution == "hourly":
        with get_cursor(SOLAR_POWER_DB_PATH) as cursor:
            query = """
                SELECT date, time
                  FROM hourly_calculated
                 ORDER BY date DESC, time DESC
                 LIMIT 1
            """
            cursor.execute(query)
            max_solar_data_date, max_solar_data_hour = cursor.fetchall()[0]
        max_solar_data_ts = datetime.strptime(max_solar_data_date + " " + max_solar_data_hour, "%Y-%m-%d %H:%M:%S")

        return max_solar_data_ts

    elif data_resolution == "daily":
        with get_cursor(SOLAR_POWER_DB_PATH) as cursor:
            query = """
                SELECT date
                  FROM monthly_calculated
                 ORDER BY date DESC
                 LIMIT 1
            """
            cursor.execute(query)
            max_solar_data_date = cursor.fetchall()[0]
        max_solar_data_ts = datetime.strptime(max_solar_data_date[0], "%Y-%m-%d")

        return max_solar_data_ts

    else:
        logging.critical(f"'{data_resolution}' is not a valid data resolution.")
        exit(1)


def get_hourly_yield_by_timestamp(date_str: str,
                                  time_str: str
                                  ) -> float | None:
    """
    Retrieves the calculated PV yield for a specific hour from the database.

    :param date_str: Date in the format ‘YYYY-MM-DD’
    :param time_str: Time in the format ‘HH:MM:SS’
    :return: PV yield (kWh) as float, or None if no value is available
    """
    query = """
      SELECT pv_yield_calc
        FROM hourly_calculated
       WHERE date = ? AND time = ?
    """
    with sqlite3.connect(SOLAR_POWER_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=[date_str, time_str])

    return float(df.iloc[0, 0]) if not df.empty else None


def get_daily_yield_by_timestamp(date_str: str) -> float | None:
    """
    Retrieves the daily PV yield from the database.

    :param date_str: Date in the format ‘YYYY-MM-DD’
    :return: PV yield (kWh) as float, or None if no value is available
    """
    query = """
      SELECT pv_yield
        FROM monthly_calculated
       WHERE date = ?
    """
    with sqlite3.connect(SOLAR_POWER_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=[date_str])

    return float(df.iloc[0, 0]) if not df.empty else None


def get_hourly_yield() -> pd.DataFrame:
    """
    Provides all available hourly yield data.

    :return: Data frame containing all available hourly yield data.
    """
    query = """
      SELECT date, time, pv_yield_calc
        FROM hourly_calculated
    """
    with sqlite3.connect(SOLAR_POWER_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def get_daily_yield() -> pd.DataFrame:
    """
    Provides all available daily yield data.

    :return: Data frame containing all available daily yield data.
    """
    query = """
      SELECT date, pv_yield
        FROM monthly_calculated
    """
    with sqlite3.connect(SOLAR_POWER_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


# ----------------------------
# Setter
# ----------------------------


if __name__ == "__main__":
    pass
