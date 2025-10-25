import sqlite3
from datetime import datetime, timedelta

import numpy as np
import openmeteo_requests

import requests_cache
from retry_requests import retry


def main():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    variables = ["temperature_2m", "precipitation", "rain", "weather_code", "cloud_cover", "cloud_cover_mid",
                 "cloud_cover_low", "cloud_cover_high", "wind_speed_10m", "relative_humidity_2m", "dew_point_2m",
                 "apparent_temperature", "precipitation_probability", "showers", "snowfall", "snow_depth",
                 "pressure_msl", "surface_pressure", "visibility", "evapotranspiration", "et0_fao_evapotranspiration",
                 "vapour_pressure_deficit", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
                 "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
                 "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm",
                 "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm",
                 "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
                 "soil_moisture_27_to_81cm"]

    params = {
        "latitude": None,
        "longitude": None,
        "hourly": variables,
        "forecast_days": 16
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()

    # Params
    db_path = "weather_forecasts.sqlite"

    now = datetime.now()
    response_date = now.date().isoformat()
    response_hour = now.time().replace(minute=0, second=0, microsecond=0).strftime("%H:%M:%S")
    # start_hour -> Forecasts begin the next day at midnight
    start_hour = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Collect data for insert statements. Key = date + time: value = list with response-values
    insert_dict = {}

    # Loop through all variables
    for i, table_name in enumerate(variables):
        all_values = hourly.Variables(i).ValuesAsNumpy()

        # Loop through all hourly values
        for hour_offset, hourly_value in enumerate(all_values):
            forecast_datetime = start_hour + timedelta(hours=hour_offset)
            forecast_date = forecast_datetime.strftime("%Y-%m-%d")
            forecast_hour = forecast_datetime.strftime("%H:%M:%S")

            # Create dictionary entry, if not exist yet
            insert_dict.setdefault(f"{response_date}_{response_hour}_{forecast_date}_{forecast_hour}",
                                   [response_date, response_hour, forecast_date, forecast_hour])

            # Handle value
            if np.isnan(hourly_value):
                value = "Null"
            else:
                value = round(float(hourly_value), 1)

            # Append value
            insert_dict[f"{response_date}_{response_hour}_{forecast_date}_{forecast_hour}"].append(value)

    # Create insert statement
    columns = "response_date, response_hour, forecast_date, forecast_hour"
    for variable in variables:
        columns += f", {variable}"
    placeholders = ", ".join(["?"] * (len(variables) + 4))
    insert_statement = f"INSERT INTO forecast ({columns}) VALUES ({placeholders})"

    # Insert rows
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for values in insert_dict.values():
            cursor.execute(insert_statement, values)

    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - Daten erfolgreich abgerufen.")


if __name__ == "__main__":
    main()
