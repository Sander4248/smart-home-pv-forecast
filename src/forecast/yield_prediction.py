from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Union, Tuple, List

import pandas as pd
from pandas import DataFrame
from xgboost import XGBModel

from scripts.db_rebuild_solpos_data import get_solpos_data
from src.clearsky.calc import clearsky_hourly, clearsky_daily
from src.config.schema import RunCfg
from src.data.weather_history_dao import get_historical_weather_data_by_timestamp
from src.data.weather_forecasts_dao import get_forecast_weather_data_by_timestamp
from src.data.solpos_data_dao import get_solpos_by_timestamp
from src.data.solar_power_dao import get_hourly_yield_by_timestamp, get_daily_yield_by_timestamp
from src.data.training_results_dao import load_model_from_db
from src.utils import append_time_features, append_date_features


def _predict_forecast_yield(forecast_df: pd.DataFrame,
                            model: XGBModel
                            ) -> float:
    """
    Calculates the predicted PV yield using a transferred model.

    :param forecast_df: DataFrame with the input variables (features) required for the forecast
    :param model: Trained model that supports a 'predict' method
    :return: Forecasted PV yield (rounded to one decimal place)
    """

    y_hat = model.predict(forecast_df)
    return round(y_hat[0], 1)


def _get_solpos_df(forecast_date: str,
                   forecast_hour: str
                   ) -> pd.DataFrame:
    """
    Returns the solar position data for the timestamp provided.

    If no data is stored in the database, the data is retrieved via the API.

    :param forecast_date: Target date in the format 'YYYY-MM-DD'
    :param forecast_hour: Target time in the format 'HH:MM:SS'
    :return: DataFrame with the sun position data for the requested date and time
    """
    df = get_solpos_by_timestamp(forecast_date, forecast_hour)
    if df.empty:
        df = _calc_solpos_df(forecast_date, forecast_hour)
    return df


def _calc_solpos_df(forecast_date: str,
                    forecast_hour: str
                    ) -> pd.DataFrame:
    """
    Retrieves solar position data via the API and filters it for a specific date and time.

    :param forecast_date: Target date in the format 'YYYY-MM-DD'
    :param forecast_hour: Target time in the format 'HH:MM:SS'
    :return: DataFrame with the sun position data for the requested date and time
    """
    df = get_solpos_data(forecast_date, forecast_date + " 23:00:00")
    df = df.loc[df.index == f"{forecast_date} {forecast_hour}"]

    # Remove useless columns and index
    df = df.drop(columns="timestamp")
    df = df.reset_index(drop=True)

    return df


def _get_forecast_df(forecast_ts: datetime,
                     evaluate_with_historical_data: bool,
                     response_ts: Union[datetime, None] = None,
                     forecast_horizon: Union[int, None] = None
                     ) -> tuple[DataFrame, str, str]:
    """
    Loads weather forecast data from the SQLite database for a specific target date and time.

    If no response time is passed:
     - Attempts to load the forecast for the given forecast horizon (in days) and the specified 'timestamp'
     - If no data is available for the calculated response time, the system goes back hour by hour ('hours_offset')
       until an entry is found in the database

    If a response time is passen:
     - It is assumed that the response and forecast times have been checked for availability in advance
     - No forecast horizon is required and no nearby response time is determined if the transferred response
       time is not available

    If the response time and forecast horizon are none, the program is terminated.
    The same applies if the response time transferred is not available in the database.

    :param forecast_horizon: Forecast horizon in days
    :param evaluate_with_historical_data: If enabled, historical instead forecast weather data is used for evaluation
    :param forecast_ts: Time for which the forecast should be retrieved
    :param response_ts: Time of the response of the forecast.
    :return: DF with the loaded forecast data and additional, forecast date ('YYYY-MM-DD') and forecast time ('HH:MM:SS')
    """
    if forecast_horizon is None and response_ts is None:
        logging.critical(f"No forecast horizon and no response timestamp were passed. "
                         f"Unable to calculate response timestamp for forecast timestamp '{forecast_ts}'."
                         f"Program is canceled!")
        exit(1)

    forecast_date = forecast_ts.strftime("%Y-%m-%d")
    forecast_hour = forecast_ts.strftime("%H:%M:%S")

    hours_offset = 0

    if response_ts is not None:
        response_date = response_ts.strftime("%Y-%m-%d")
        response_hour = response_ts.strftime("%H:%M:%S")

        df = _get_weather_forecast_data(forecast_date, forecast_hour, response_date, response_hour,
                                        evaluate_with_historical_data)

    else:
        # Search until forecast data is available (go back hour by hour)
        df = pd.DataFrame()
        while df.empty:
            # Params for forecast entry
            response_ts_calc = forecast_ts - timedelta(hours=forecast_horizon * 24 + hours_offset)
            response_date = response_ts_calc.strftime("%Y-%m-%d")
            response_hour = response_ts_calc.strftime("%H:%M:%S")

            df = _get_weather_forecast_data(forecast_date, forecast_hour, response_date, response_hour,
                                            evaluate_with_historical_data)

            hours_offset += 1

    # Convert the string date and string time to a panda.datetime
    df["datetime"] = pd.to_datetime(df["forecast_date"] + " " + df["forecast_hour"])

    return df, forecast_date, forecast_hour


def _get_weather_forecast_data(forecast_date: str,
                               forecast_hour: str,
                               response_date: str,
                               response_hour: str,
                               evaluate_with_historical_data: bool
                               ) -> pd.DataFrame:
    """
    Retrieves weather forecast data from an SQLite database. This is the standard case.

    Attention:
    It is also possible to load historical data instead of forecast data.
    However, this is only for testing purposes!
    This is set via the variable evaluate_with_historical_data.

    :param forecast_date: Date of the weather forecast, format “%Y-%m-%d”
    :param forecast_hour: Time of the weather forecast, format “%H:%M:%S”
    :param response_date: Date of the response, format “%Y-%m-%d”
    :param response_hour: Time of the response, format “%H:%M:%S”
    :param evaluate_with_historical_data: If enabled, historical instead forecast weather data is used for evaluation
    :return: pandas.DataFrame with the weather forecast. If no data matches, an empty DataFrame is returned.
    """
    if evaluate_with_historical_data:
        df = get_historical_weather_data_by_timestamp(forecast_date, forecast_hour)

        df = df.rename(columns={"date": "forecast_date"})
        df = df.rename(columns={"hour": "forecast_hour"})

    else:
        df = get_forecast_weather_data_by_timestamp(forecast_date, forecast_hour, response_date, response_hour)

    return df


def _aggregate_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weather data by date using specific rules.
    Automatically considers all numeric features in the DataFrame.

    :param df: Pandas dataframe containing forecast data
    :return: Pandas dataframe containing aggregated forecast data
    """

    # Special rules
    special_aggs = {
        "precipitation": "sum",
        "rain": "sum",
        "snowfall": "sum",
        "snow_depth": "max",
        "wind_gusts_10m": "max",
        "et0_fao_evapotranspiration": "sum",
        "apparent_elevation": "max",  # sun's zenith
        "apparent_zenith": "min"  # lowest angle to the zenith (highest sun)
    }

    # Remove columns
    drop_cols = ["response_hour", "response_date", "forecast_hour"]
    df_clean = df.drop(columns=drop_cols, errors="ignore")

    # Rename 'forecast_date' to 'date'
    df_clean = df_clean.rename(columns={"forecast_date": "date"})

    # Defaultdict with 'mean'
    agg_dict = defaultdict(lambda: "mean")

    # Use all numeric columns
    numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()

    for feature in numeric_cols:
        agg_dict[feature] = special_aggs.get(feature, "mean")

    # Perform aggregation
    df_agg = df_clean.groupby("date", as_index=False).agg({col: agg_dict[col] for col in numeric_cols})

    return df_agg


def _maybe_denormalize_hourly(cfg: RunCfg, forecast_ts: datetime, y_hat: float) -> float:
    """
    Optionally denormalize an hourly forecast value if clear-sky normalization was used.

    :param cfg: Central configuration instance for the training run with all parameters
    :param forecast_ts: Forecast timestamp (local, naive or tz-aware)
    :param y_hat: Predicted (normalized) PV yield value
    :return: Denormalized PV yield in kWh if normalization was active. Otherwise, returns 'y_hat' unchanged.
    """
    if cfg.training.normalize_by_clearsky:
        # Get the clear-sky reference value for this timestamp
        cs = clearsky_hourly(cfg, pd.DatetimeIndex([pd.Timestamp(forecast_ts)]))
        p_cs = float(cs["P_cs_kWh"].iloc[0])

        # If clear-sky value is too small → treat as zero
        if p_cs < cfg.params.eps_pcs:
            return 0.0

        # Scale normalized prediction back to real-world kWh
        return round(float(y_hat) * p_cs, 1)

    # No normalization active → return unchanged prediction
    return y_hat


def _maybe_denormalize_daily(cfg: RunCfg, forecast_date: pd.Timestamp, y_hat: float) -> float:
    """
    Optionally denormalize a daily forecast value if clear-sky normalization was used.

    :param cfg: Central configuration instance for the training run with all parameters
    :param forecast_date: Forecast date (local-naive)
    :param y_hat: redicted (normalized) PV yield value.
    :return: Denormalized PV yield in kWh if normalization was active. Otherwise, returns 'y_hat' unchanged.
    """
    if cfg.training.normalize_by_clearsky:
        # Get the clear-sky reference value for this date
        cs = clearsky_daily(cfg, forecast_date)
        p_cs = float(cs["P_cs_kWh"].iloc[0])

        # If clear-sky value is too small → treat as zero
        if p_cs < cfg.params.eps_pcs:
            return 0.0

        # Scale normalized prediction back to real-world kWh
        return round(float(y_hat) * p_cs, 1)

    # No normalization active → return unchanged prediction
    return y_hat


def predict_hourly_yield(cfg: RunCfg,
                         forecast_ts: datetime,
                         response_ts: datetime = None,
                         result_id: int = None,
                         model: XGBModel = None
                         ) -> tuple[float, float, float]:
    """
    Main function for calculating the predicted and actual PV yield for a given point in time.
     - Creates a DataFrame with weather forecast data, sun position data, and time and date data
     - Selects the features used for model training
     - Forecasts the PV yield and retrieves the actual yield from the DB

    Model:
     - If a 'result_id' is passed, the corresponding trained model is loaded from the DB
     - If no model and no 'result_id' are specified, the program terminates with an error
     - One of the two parameters, result_id or model, must be passed, otherwise the program will terminate

    Forecast-horizon/Response-timestamp
     - If a response timestamp is passed, it takes precedence in the _get_forecast_df function
     - Otherwise, the response timestamp is determined with the forecast timestamp and the forecast
       horizon in the _get_forecast_df function
     - One of the two parameters, response_ts or forecast_horizon, must be passed, otherwise the program will terminate

    :param cfg: Central configuration instance for the training run with all parameters
    :param forecast_ts: Time for which the PV yield is to be calculated
    :param response_ts: optional.
    :param result_id: optional. ID of a model stored in the DB. If set, the model is loaded from the DB
    :param model: optional. Model that has already been loaded or transferred externally.
                            Only used if no 'result_id' is specified.
    :return: Forecasted PV yield and Actual measured PV yield (Both in kWh and rounded to one decimal place)
             and apparent elevation for night clamp
    """
    if result_id is not None:
        # Get trained model from DB
        model = load_model_from_db(cfg.training.data_resolution, result_id)
    elif model is None:
        logging.critical("No model and no result_id passed. Program is canceled!")
        exit(1)

    # Get dataframe with forecast features
    if response_ts:
        # Response timestamp is transferred directly
        forecast_df, forecast_date, forecast_hour = (
            _get_forecast_df(forecast_ts, cfg.evaluation.eval_with_weather_history_data, response_ts=response_ts))
    else:
        # Determines the response time stamp from the forecast time stamp and the forecast horizon
        forecast_df, forecast_date, forecast_hour = (
            _get_forecast_df(forecast_ts, cfg.evaluation.eval_with_weather_history_data,
                             forecast_horizon=cfg.evaluation.forecast_horizon))

    # Add solar position features to dataframe
    solpos_df = _get_solpos_df(forecast_date, forecast_hour)
    df = pd.concat([forecast_df, solpos_df], axis=1)

    # Add datetime features to dataframe
    df = append_time_features(df, "datetime")
    df = append_date_features(df, "datetime")

    # Add clear sky feature/s to dataframe
    cs = clearsky_hourly(cfg, pd.DatetimeIndex([pd.Timestamp(forecast_ts)]))
    df = pd.concat([df, cs], axis=1)

    # Get features, wich are used for model training from model
    features = model.feature_names_in_

    # Collect necessary columns for model evaluation
    forecast_df = df[features]

    # Retrieve forecast and actual yields and calculate deviations
    forecast_yield = _predict_forecast_yield(forecast_df, model)
    forecast_yield = _maybe_denormalize_hourly(cfg, pd.to_datetime(forecast_date + " " + forecast_hour), forecast_yield)
    actual_yield = get_hourly_yield_by_timestamp(forecast_date, forecast_hour)

    # Get 'apparent_elevation' from solpos if available
    if "apparent_elevation" in solpos_df.columns and not solpos_df["apparent_elevation"].empty:
        apparent_elevation = solpos_df["apparent_elevation"].iloc[0]
    else:
        apparent_elevation = None

    return forecast_yield, actual_yield, apparent_elevation


def predict_daily_yield(cfg: RunCfg,
                        daily_timestamps: List[Tuple[datetime, datetime]],
                        result_id: int = None,
                        model: XGBModel = None
                        ) -> tuple[float, float]:
    """
    Main function for calculating the predicted and actual PV yield for a given day.
     - Creates a DataFrame with weather forecast data, sun position data, and time and date data
     - Selects the features used for model training
     - Forecasts the PV yield and retrieves the actual yield from the DB

    Model:
     - If a 'result_id' is passed, the corresponding trained model is loaded from the DB
     - If no model and no 'result_id' are specified, the program terminates with an error
     - One of the two parameters, result_id or model, must be passed, otherwise the program will terminate

    Forecast-horizon/Response-timestamp
     - If a response timestamp is passed, it takes precedence in the _get_forecast_df function
     - Otherwise, the response timestamp is determined with the forecast timestamp and the forecast
       horizon in the _get_forecast_df function
     - One of the two parameters, response_ts or forecast_horizon, must be passed, otherwise the program will terminate

    :param cfg: Central configuration instance for the training run with all parameters
    :param daily_timestamps: List with 24 Tuples(response timestamp, forecast timestamp)
    :param result_id: optional. ID of a model stored in the DB. If set, the model is loaded from the DB
    :param model: optional. Model that has already been loaded or transferred externally.
                            Only used if no 'result_id' is specified.
    :return: Forecasted PV yield and Actual measured PV yield (Both in kWh and rounded to one decimal place)
    """
    if result_id is not None:
        # Get trained model from DB
        model = load_model_from_db(cfg.training.data_resolution, result_id)
    elif model is None:
        logging.critical("No model and no result_id passed. Program is canceled!")
        exit(1)

    df = pd.DataFrame()
    for response_ts, forecast_ts in daily_timestamps:
        # Get dataframe with forecast features
        forecast_df, forecast_date, forecast_hour = (
            _get_forecast_df(forecast_ts, cfg.evaluation.eval_with_weather_history_data ,response_ts=response_ts))

        # Add solar position features to dataframe
        solpos_df = _get_solpos_df(forecast_date, forecast_hour)
        row = pd.concat([forecast_df, solpos_df], axis=1)
        df = pd.concat([df, row], ignore_index=True)

    df = _aggregate_forecast(df)
    # Convert string date to datetime64
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    date = df["date"]

    # Add datetime features to dataframe
    df = append_date_features(df, "date")

    # Add clear sky feature/s to dataframe
    cs = clearsky_daily(cfg, df["date"])
    df = pd.concat([df, cs], axis=1)

    # Get features, wich are used for model training from model
    features = model.feature_names_in_

    # Collect necessary columns for model evaluation
    forecast_df = df[features]

    # Retrieve forecast and actual yields and calculate deviations
    forecast_yield = _predict_forecast_yield(forecast_df, model)
    forecast_yield = _maybe_denormalize_daily(cfg, pd.to_datetime(date), forecast_yield)
    actual_yield = get_daily_yield_by_timestamp(forecast_date)

    return forecast_yield, actual_yield



