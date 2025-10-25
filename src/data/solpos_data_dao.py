from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from src.config.paths import SOLPOS_DATA_DB_PATH

"""
Important:
'timestamp'  contains values with time zone information. Since the records from other databases are stored
as local time without time zone, the time zone information (+02:00) is truncated.
Result: uniform display of all timestamps in local time without offset.
"""


# ----------------------------
# Getter
# ----------------------------

def get_solpos_in_range(months: str,
                        features: str
                        ) -> pd.DataFrame:
    """
    Loads the solar position data for selected months and features from the database.

    If all data is to be loaded, all months (as numbers) must be entered in the ‘months_str’ parameter.
    All data for the filtered months is output regardless of the year.

    'timestamp'  contains values with time zone information. Since the records from other databases are stored
    as local time without time zone, the time zone information (+02:00) is truncated.
    Result: uniform display of all timestamps in local time without offset.

    :param months: Comma separated string of months e.g. '1, 2, 3'
    :param features: Comma separated string of features (columns) to be loaded
    :return: DataFrame with the sun position data for the requested months
    """
    query = f"""
         select substr(timestamp, 1, 19) as datetime, {features}
          from solar_position
         where strftime('%m', datetime) + 0  in ({months})
    """

    with sqlite3.connect(SOLPOS_DATA_DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def get_solpos_by_timestamp(date: str,
                            hour: str
                            ) -> pd.DataFrame:
    """
    Load solar position data from BD for a specific date and time.

    :param date: Target date in the format 'YYYY-MM-DD'
    :param hour: Target time in the format 'HH:MM:SS'
    :return: DataFrame with the sun position data for the requested date and time
    """
    query = f"""
        SELECT apparent_elevation, apparent_zenith, azimuth
          FROM solar_position
          WHERE substr(timestamp, 1, 19) = datetime('{date}' || ' ' || '{hour}');
    """
    with sqlite3.connect(SOLPOS_DATA_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    if len(df) > 1:
        logging.warning(f"More then one entry for solar position in Database! "
                        f"Entries: {len(df)}, Date: {date}, Hour: {hour}")

    return df


def get_apparent_elevation() -> pd.DataFrame:
    """
    Determines all hourly apparent elevation data available and the corresponding timestamp.



    :return: DataFrame with data
    """
    query = f"""
        SELECT substr(timestamp, 1, 19) as datetime, apparent_elevation
          FROM solar_position;
    """
    with sqlite3.connect(SOLPOS_DATA_DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


# ----------------------------
# Setter
# ----------------------------

if __name__ == "__main__":
    pass
