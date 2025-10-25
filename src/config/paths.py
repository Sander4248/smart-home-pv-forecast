from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
SOLAR_POWER_DB_PATH = DATA_DIR / 'solar_power.db'
SOLPOS_DATA_DB_PATH = DATA_DIR / 'solpos_data.db'
RESULTS_DB_PATH = DATA_DIR / 'training_results.db'
WEATHER_FORECAST_DB_PATH = DATA_DIR / 'weather_forecasts.sqlite'
WEATHER_HISTORY_DB_PATH = DATA_DIR / 'weather_history.db'
