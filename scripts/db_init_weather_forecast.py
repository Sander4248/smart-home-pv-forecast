import logging
import os
import sqlite3

from src.config.logging_config import setup_logging
from src.config.paths import WEATHER_FORECAST_DB_PATH


def main():
    """
    Creates the database ‘weather_forecasts’.

    All weather features are listed in a table.

    If the database already exists, it will be deleted and recreated.
    """

    setup_logging()

    # Questions about whether the database should be reset
    if os.path.exists(WEATHER_FORECAST_DB_PATH):
        confirm = input(f"Database '{WEATHER_FORECAST_DB_PATH}' already exists. Really want to delete? "
                        f"The data cannot be reproduced because it was collected using a cron job. Delete? (yes/no): ")

        # Only delete if “yes” has been explicitly entered
        if confirm.strip().lower() == "yes":
            os.remove(WEATHER_FORECAST_DB_PATH)
            logging.warning(f"Deleted database '{WEATHER_FORECAST_DB_PATH}'")
        else:
            logging.info(f"Aborted. Database '{WEATHER_FORECAST_DB_PATH}' was not deleted.")
            exit(1)

    # List of parameter names = table names
    table_names = [
        "temperature_2m", "precipitation", "rain", "weather_code", "cloud_cover",
        "cloud_cover_mid", "cloud_cover_low", "cloud_cover_high", "wind_speed_10m",
        "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability",
        "showers", "snowfall", "snow_depth", "pressure_msl", "surface_pressure", "visibility",
        "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
        "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m",
        "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m",
        "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm",
        "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm",
        "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm",
        "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"
    ]

    # Create 'create table' statement
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS forecast (
            response_date DATE NOT NULL,
            response_hour TIME NOT NULL,
            forecast_date DATE NOT NULL,
            forecast_hour TIME NOT NULL,
            """
    for column in table_names:
        create_sql += f"{column} REAL, "
    create_sql += f"UNIQUE(response_date, response_hour, forecast_date, forecast_hour))"

    # Create database and table.
    with sqlite3.connect(WEATHER_FORECAST_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(create_sql)

    logging.info(f"Database '{WEATHER_FORECAST_DB_PATH}' successfully created")


if __name__ == "__main__":
    main()
