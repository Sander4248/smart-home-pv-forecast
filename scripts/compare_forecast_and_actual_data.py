import logging
import sqlite3
import re
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config.logging_config import setup_logging
from src.config.paths import WEATHER_HISTORY_DB_PATH, WEATHER_FORECAST_DB_PATH
from src.utils import get_cursor


def _coerce_numeric_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and converts forecast/actual columns to numeric values.

    Steps:
      - Finds all columns with the suffix “_forecast” and “_actual”
      - Replaces commas with periods and removes whitespace
      - Converts values to numeric (float); unreadable values become NaN

    :param df: DataFrame with forecast and actual column
    :return: Copy of the DataFrame with numerically cleaned forecast and actual columns
    """
    # Find all forecast/actual columns
    f_cols = [c for c in df.columns if c.endswith("_forecast")]
    a_cols = [c for c in df.columns if c.endswith("_actual")]
    all_cols = f_cols + a_cols

    # Commas -> periods, trim whitespace
    df[all_cols] = (df[all_cols]
                    .apply(lambda s: s.astype(str).str.replace(",", ".", regex=False).str.strip()))

    # Convert to numbers; unreadable -> NaN
    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates error metrics to evaluate forecasts in comparison to historical values.

    Procedure:
      - Converts forecast/actual columns to numeric values (_coerce_numeric_pairs)
      - Identifies feature base names (e.g., “temperature” from “temperature_forecast/actual”)
      - Calculates for each feature:
        * n: Number of valid value pairs
        * MAE: Mean absolute error
        * RMSE: Root mean square error
        * Bias: Average deviation (forecast - actual)
        * Corr: Correlation coefficient between forecast and actual

    :param df: DataFrame with forecast and actual columns.
    :return: DataFrame with error metrics, indexed by feature name, sorted by correlation coefficient.
    """
    # Use a copy so as not to change the original DataFrame
    df = _coerce_numeric_pairs(df.copy())

    # Extract base names (without _forecast/_actual)
    bases = sorted({re.sub(r"_forecast$", "", c) for c in df.columns if c.endswith("_forecast")})
    rows = []

    # Loop over all feature base names (e.g., “temperature”)
    for base in bases:
        fcol = f"{base}_forecast"
        acol = f"{base}_actual"

        # If either forecast or actual is missing -> skip
        if fcol not in df or acol not in df:
            continue

        yhat = df[fcol]  # Forecast values
        y = df[acol]  # Actual values
        mask = yhat.notna() & y.notna()  # Only consider pairs with valid values
        n = int(mask.sum())  # Number of valid value pairs

        # If no valid data points -> empty row with NaN values
        if n == 0:
            rows.append({"feature": base, "n": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Corr": np.nan})
            continue

        # Calculate errors (forecast - actual) and standard error metrics
        e = (yhat[mask] - y[mask])
        mae = e.abs().mean()
        rmse = np.sqrt((e ** 2).mean())
        bias = e.mean()

        # Correlation coefficient (if variances > 0 and sufficient data are available)
        r = np.corrcoef(yhat[mask], y[mask])[0, 1] if n > 1 and yhat[mask].std() > 0 and y[mask].std() > 0 else np.nan

        # Collect results for this feature
        rows.append({"feature": base, "n": n, "MAE": mae, "RMSE": rmse, "Bias": bias, "Corr": r})

    # Return results as a DataFrame and sort by correlation
    out = pd.DataFrame(rows).set_index("feature").sort_values(["Corr"])

    return out


def main(forecast_horizon: int = 1):
    """
    Performs a comparison between weather forecast data and historical weather data to evaluate forecast quality
    using error metrics (MAE, RMSE, bias, correlation).

    Process:
      1. Determines the earliest available forecast time in the forecast database.
      2. Loads historical weather data from this point in time.
      3. Creates timestamps and searches the forecast database for forecasts that correspond to the historical
         times (using a backward search in the horizon if necessary).
      4. Links forecast and actual data via common timestamps.
      5. Calculates metrics to evaluate the forecast quality.
      6. Logs results in the log.

    :param forecast_horizon: Determines how many days before the forecast date the forecast database is searched
    :return: None: Results (metrics) are written to the lo
    """
    setup_logging()
    logging.info("Starting forecast comparison")

    # Get minimal forecast response timestamp
    query = """
        SELECT MIN(response_date)
          FROM forecast
    """
    with get_cursor(WEATHER_FORECAST_DB_PATH) as cursor:
        cursor.execute(query)
        min_date = cursor.fetchone()[0]

    # Get historical weather data from the database starting from the point in time at which forecast data is available
    query = f"""
        SELECT *
          FROM history
         WHERE date >= '{min_date}'
    """

    with sqlite3.connect(WEATHER_HISTORY_DB_PATH) as conn:
        df_actual = pd.read_sql_query(query, conn)

    # Merge columns and convert to datetime
    df_actual["datetime"] = pd.to_datetime(df_actual["date"] + " " + df_actual["hour"])

    # In a Python list of datetime objects
    dt_list = df_actual["datetime"].tolist()

    # Search the forecast database for all dates that appear in the history database.
    rows = []
    with sqlite3.connect(WEATHER_FORECAST_DB_PATH) as conn:
        for date in dt_list:
            hours_offset = 0
            df = pd.DataFrame()
            forecast_date = date.strftime("%Y-%m-%d")
            forecast_hour = date.strftime("%H:%M:%S")

            # Search until forecast data is available (go back hour by hour)
            while df.empty:

                # Params for forecast entry
                response_ts_calc = date - timedelta(hours=forecast_horizon * 24 + hours_offset)
                response_date = response_ts_calc.strftime("%Y-%m-%d")
                response_hour = response_ts_calc.strftime("%H:%M:%S")

                # Exit the while loop if the date looking for is less than the minimum available date
                if response_date < min_date:
                    break

                # Load searched timestamp from database
                query = f"""
                        SELECT *
                          FROM forecast
                         WHERE response_date = '{response_date}'
                           AND response_hour = '{response_hour}'
                           AND forecast_date = '{forecast_date}'
                           AND forecast_hour = '{forecast_hour}'
                    """
                df = pd.read_sql_query(query, conn)

                # Increase the offset by 1 hour (if df is empty, the next entry is searched for with this offset)
                hours_offset += 1

            if not df.empty:
                rows.append(df)

    # Concat all rows into a DataFrame
    df_forecast = pd.concat(rows)

    # Create timestamp from date and hour
    df_actual["timestamp"] = pd.to_datetime(df_actual["date"]) + pd.to_timedelta(df_actual["hour"])
    df_forecast["timestamp"] = pd.to_datetime(df_forecast["forecast_date"]) + pd.to_timedelta(
        df_forecast["forecast_hour"])

    # Create datetime index
    df_actual = df_actual.set_index("timestamp").sort_index()
    df_forecast = df_forecast.set_index("timestamp").sort_index()

    # Delete columns that are not needed
    df_actual = df_actual.drop(columns=['date', 'hour', 'datetime'])
    df_forecast = df_forecast.drop(columns=['response_date', 'response_hour', 'forecast_date', 'forecast_hour'])

    # Merge dataframes using DatetimeIndex
    aligned = df_forecast.join(df_actual, how="inner", lsuffix="_forecast", rsuffix="_actual")

    metrics = _compute_metrics(aligned)

    logging.info(metrics)
    logging.info("Ending forecast comparison")


if __name__ == "__main__":
    main()
