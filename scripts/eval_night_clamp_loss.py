import logging

import pandas as pd

from src.config.logging_config import setup_logging
from src.data.solar_power_dao import get_hourly_yield
from src.data.solpos_data_dao import get_apparent_elevation


def evaluate_night_clamping():
    """
    Calculates the effect of night clamping on PV yield forecasts.

    This function loads hourly PV yield data and apparent solar altitude from the database, links them via uniform
    timestamps, and sets negative or unrealistic yield values during the night (below a defined solar altitude) to
    zero. It then calculates how much energy has been 'removed' as a result and what percentage of the total forecast
    yield this represents. The results are written to the log.

    Steps:
        1. Load PV yield data and sun heights.
        2. Standardize timestamps and perform an inner join.
        3. Set all values with sun height <= 0 to zero (night clamping).
        4. Calculate the difference and the percentage of energy removed.
        5. Log the results.
    """
    setup_logging()

    # Threshold value used for nighttime clamping. Predictions below this threshold during night are adjusted.
    night_clamp_threshold = 0

    # Load required data records from database
    df_yield = get_hourly_yield()
    df_solpos = get_apparent_elevation()

    # Create uniform timestamps to link data records
    df_solpos["datetime"] = pd.to_datetime(df_solpos["datetime"], format="%Y-%m-%d %H:%M:%S")
    df_yield["datetime"] = pd.to_datetime(df_yield["date"] + " " + df_yield["time"], format="%Y-%m-%d %H:%M:%S")

    # Delete unnecessary columns
    df_yield.drop(columns=["date", "time"], inplace=True)

    # Inner join of the records
    df = pd.merge(df_yield, df_solpos, on="datetime", how="inner")

    # Apply night clamp
    df["pv_yield_calc_clamped"] = df["pv_yield_calc"].where(df["apparent_elevation"] > night_clamp_threshold, 0)

    # Calculate 'clamped' kWh
    df["clamped_diff"] = df["pv_yield_calc"] - df["pv_yield_calc_clamped"]
    total_removed = df["clamped_diff"].sum()

    # Percentage relative to the total sum of the unclamped prediction
    pct_removed = 100 * total_removed / df["pv_yield_calc"].sum()

    logging.info(f"Clamped energy in kWh: {total_removed:.2f} kWh")
    logging.info(f"Share in percent: {pct_removed:.2f} %")

if __name__ == "__main__":
    evaluate_night_clamping()