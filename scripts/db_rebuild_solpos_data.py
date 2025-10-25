import sqlite3
from pvlib.location import Location
import pandas as pd

from src.config.paths import SOLPOS_DATA_DB_PATH
from src.config.schema import ProjectParams


def get_solpos_data(start = None, end = None) -> pd.DataFrame:
    """
    Calculates hourly solar position data for a fixed time range.

     - The function uses 'pvlib' to calculate solar position for each hour during a defined period of time.
     - Only useful columns are kept, and the timestamp is added as a separate column.
     - The resulting data is stored in a SQLite table named 'solar_position'.
    :return: DataFrame with hourly solar position data
    """

    cfg = ProjectParams()

    if start is None:
        start = cfg.START_DATE
    if end is None:
        end = cfg.end_date

    # Location definition
    site = Location(cfg.LATITUDE, cfg.LONGITUDE, cfg.TIMEZONE)

    # Define time series
    times = pd.date_range(start, end, freq='1h', tz=cfg.TIMEZONE)

    # Get date from pvlib API
    solpos = site.get_solarposition(times)

    # Filter relevant columns and add index (timestamp) as a column
    solpos_relevant = solpos[['apparent_elevation', 'apparent_zenith', 'azimuth']].copy()

    # Timestamp as a column for SQL
    solpos_relevant['timestamp'] = solpos_relevant.index

    return solpos_relevant


if __name__ == "__main__":
    """
    Stores the relevant values in a SQLite database
    
    Note: Set start and end date in src.config.para
    """

    solpos_data = get_solpos_data()

    # Connect to the database or create a new one and save the data frame in a table
    with sqlite3.connect(SOLPOS_DATA_DB_PATH) as conn:
        solpos_data.to_sql('solar_position', conn, if_exists='replace', index=False)
