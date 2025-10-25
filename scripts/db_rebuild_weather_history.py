import logging

import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
import sqlite3
import os
import numpy as np

from src.config.logging_config import setup_logging
from src.config.paths import WEATHER_HISTORY_DB_PATH
from src.config.schema import ProjectParams


def _prepare_data(response, variables):
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    date_data = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Collect data for insert statements. Key = date + time: value = list with response-values
    insert_dict = {}

    # Loop through all variables
    for i, variable in enumerate(variables):
        all_data = hourly.Variables(i).ValuesAsNumpy()

        # Loop through all hourly values
        for j, data in enumerate(all_data):
            timestamp = date_data[j]
            date = timestamp.strftime("%Y-%m-%d")
            time = timestamp.strftime("%H:%M:%S")

            # Create dictionary entry, if not exist yet
            insert_dict.setdefault(f"{date}{time}", [date, time])

            # Handle value
            if np.isnan(data):
                value = "Null"
            else:
                value = round(float(data), 1)

            # Append value
            insert_dict[f"{date}{time}"].append(value)

    return insert_dict


def _setup_response(variables):

    cfg = ProjectParams()

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": cfg.LATITUDE,
        "longitude": cfg.LONGITUDE,
        "start_date": cfg.START_DATE,
        "end_date": cfg.end_date,
        "hourly": variables
    }
    responses = openmeteo.weather_api(url, params=params)

    return responses


def _insert_data(insert_dict, variables):
    # Create insert statement
    columns = "date, hour"
    for variable in variables:
        columns += f", {variable}"
    placeholders = ", ".join(["?"] * (len(variables) + 2))
    insert_statement = f"INSERT INTO history ({columns}) VALUES ({placeholders})"

    # Insert rows
    with sqlite3.connect(WEATHER_HISTORY_DB_PATH) as conn:
        cursor = conn.cursor()
        for values in insert_dict.values():
            cursor.execute(insert_statement, values)

    logging.info("Data entered into database")


def _setup_database(variables):
    # Erase database, if exist
    if os.path.exists(WEATHER_HISTORY_DB_PATH):
        os.remove(WEATHER_HISTORY_DB_PATH)

    # Build create statement for table (all variables in one table)
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS history (
            date DATE NOT NULL,
            hour TIME NOT NULL,
        """

    # Append columns to statement
    for name in variables:
        create_sql = f"{create_sql} {name} REAL,"

    # Append end of statement
    create_sql = f"{create_sql} UNIQUE(date, hour))"
    with sqlite3.connect(WEATHER_HISTORY_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(create_sql)

    logging.info(f"Database '{WEATHER_HISTORY_DB_PATH}' deleted and recreated.")


def main():
    """
    Stores hourly historical weather data for a specific period in a database.
     - The data is loaded via the open-meteo API.
     - The database is reset each time.
     - The weather features for which the historical data is loaded are specified under the ‘variables’ list.
    """
    setup_logging()
    variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation",
                 "rain", "snow_depth", "snowfall", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
                 "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
                 "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
                 "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
                 "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm",
                 "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"]

    # Setup database
    _setup_database(variables)

    # Setup and get response
    responses = _setup_response(variables)
    response = responses[0]
    logging.info("Response received from 'open-meteo'")

    # Prepare and insert data
    insert_dict = _prepare_data(response, variables)
    _insert_data(insert_dict, variables)
    os.remove(".cache.sqlite")

if __name__ == "__main__":
    """
    Note: Set start and end date in src.config.para
    """
    main()
