from collections import defaultdict
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from src.clearsky.calc import clearsky_daily, clearsky_hourly
from src.config.schema import RunCfg
from src.data.solpos_data_dao import get_solpos_in_range
from src.data.weather_history_dao import get_historical_weather_data_in_range
from src.data.solar_power_dao import get_hourly_yield_in_range, get_daily_yield_in_range
from src.training.utils import convert_timestamp, cast_string_values
from src.utils import append_time_features, append_date_features, create_comma_separated_string


def _get_hourly_pv_data(months: list[int]) -> pd.DataFrame:
    """
    Loads hourly PV data for the specified months and converts timestamps to the desired resolution.

    :param months: List of month numbers (e.g., [1, 2, 3]) for which to load the data.
    :return: Hourly PV data with timestamps adjusted to the specified resolution.
    """
    df = get_hourly_yield_in_range(create_comma_separated_string(months))
    df = convert_timestamp(df, "hourly")
    return df


def _get_daily_pv_data(months: list[int]) -> pd.DataFrame:
    """
    Loads daily PV data for the specified months and converts timestamps to the desired resolution.

    :param months: List of month numbers (e.g., [1, 2, 3]) for which to load the data.
    :return: Daily PV data with timestamps adjusted to the specified resolution.
    """
    df = get_daily_yield_in_range(create_comma_separated_string(months))
    df = convert_timestamp(df, "daily")
    return df


def _get_hourly_weather_data(months: list[int],
                             features: list[str],
                             limit_training_data: bool
                             ) -> pd.DataFrame:
    """
    Loads hourly weather data for specified months and features to the desired resolution.

    Casts string values to proper types and converts timestamps.

    :param months: List of month numbers (e.g., [1, 2, 3]) for which to load the data.
    :param features: List of weather features to include (e.g., ["rain", "temperature_2m"]
    :param limit_training_data: Limits the training data to the period for which no weather forecast data is available
    :return: Hourly weather data with selected features and converted timestamps.
    """
    df = get_historical_weather_data_in_range(create_comma_separated_string(months),
                                              create_comma_separated_string(features),
                                              limit_training_data)

    df = cast_string_values(df, features)
    df = convert_timestamp(df, "hourly")
    return df


def _get_daily_weather_data(months: list[int],
                            features: list[str],
                            limit_training_data: bool) -> pd.DataFrame:
    """
    Aggregates hourly weather data into daily summaries using predefined statistical methods (e.g., sum, mean, max).

    Creates a dictionary that defines the aggregations of the features.
     - These are needed to obtain meaningful data from the hourly weather data that describes the entire day.

    :param months: List of month numbers (e.g., [1, 2, 3]) for which to load the data.
    :param features: List of weather features to include (e.g., ["rain", "temperature_2m"]
    :param limit_training_data: Limits the training data to the period for which no weather forecast data is available
    :return: Daily weather data with aggregated values per day.
    """
    special_aggs = {
        "precipitation": "sum",
        "rain": "sum",
        "snowfall": "sum",
        "snow_depth": "max",
        "wind_gusts_10m": "max",
        "et0_fao_evapotranspiration": "sum"
    }

    # defaultdict with default value “mean”
    agg_dict = defaultdict(lambda: "mean")

    # Initially fill with all features
    for feature in features:
        agg_dict[feature] = special_aggs.get(feature, "mean")

    df = _get_hourly_weather_data(months, features, limit_training_data)

    # Convert datetime in date
    df['date'] = df['datetime'].dt.date

    # Aggregation of each weather feature to daily values. The aggregation functions are stored in the mapping dict
    df = df.groupby('date').agg(agg_dict).reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    return df


def _get_hourly_solar_pos_data(months: list[int], features: list[str]) -> pd.DataFrame:
    """
    Loads hourly solar position data for specified months and features, and converts timestamps to the
    required resolution.

    :param months: List of month numbers (e.g., [1, 2, 3]) for which to load the data.
    :param features: Solar position features to aggregate (e.g., "apparent_zenith").
    :return: Hourly solar position data with selected feature
    """
    df = get_solpos_in_range(create_comma_separated_string(months), create_comma_separated_string(features))
    df = convert_timestamp(df, "hourly")
    return df


def _get_daily_solar_pos_data(months: list[int], features: list[str]) -> Union[pd.DataFrame, None]:
    """
    Aggregates hourly solar position data into daily summaries using specific aggregation methods for solar features.

    Creates a dictionary that defines the aggregations of the features.
     - These are needed to obtain meaningful data from the hourly solar position data that describes the entire day.

    :param months: List of month numbers (e.g., [1, 2, 3]) for which to load the data.
    :param features: Solar position features to aggregate (e.g., "apparent_zenith").
    :return: Daily solar position data with aggregated values, or None if no aggregatable features are provided.
    """
    special_aggs = {
        "apparent_elevation": "max",  # sun's zenith
        "apparent_zenith": "min"  # lowest angle to the zenith (highest sun)
    }

    # Initially fill with all features
    agg_dict = {}
    for feature in features:
        if feature in special_aggs:
            agg_dict[feature] = special_aggs.get(feature)

    if agg_dict:
        df = _get_hourly_solar_pos_data(months, features)

        # Convert datetime in date
        df['date'] = df['datetime'].dt.date

        # Aggregation of each weather feature to daily values. The aggregation functions are stored in the mapping dict
        df = df.groupby('date').agg(agg_dict).reset_index()
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        return df

    else:
        return None


def build_dataframe(cfg: RunCfg,
                    months: int | List[int]
                    ) -> pd.DataFrame:
    """
    Builds a comprehensive dataset by combining PV data with optional weather, solar position, and
    time-based features at a specified resolution.

    Supported, optional feature keys:
     - 'weather': list of weather features
     - 'solar_pos': list of solar position features
     - 'timestamp': boolean indicating whether to add time-based features

    :param cfg: Central configuration instance for the training run with all parameters
    :param months: Month or list of months to include in the dataset.
    :return: Combined and processed dataset ready for training or analysis.
    """
    # Normalize single month to list to simplify downstream logic
    if type(months) == int:
        months = [months]

    # Route to resolution-specific builder
    if cfg.training.data_resolution == 'hourly':
        df = _build_hourly_database(cfg, months)
    else:  # daily
        df = _build_daily_dataframe(cfg, months)

    return df


def _build_daily_dataframe(cfg: RunCfg,
                           months: int | List[int],
                           ) -> pd.DataFrame:
    """
    Assemble the daily-resolution dataframe: PV → (optional) weather → (optional) solar-pos
    → (optional) timestamp features → (optional) clear-sky feature → (optional) normalization.

    :param cfg: Central configuration instance for the training run with all parameters
    :param months: Months to include
    :return: Hourly dataframe with requested features. If normalization is enabled, includes 'pv_norm'
    """
    # Base PV target at daily resolution
    df = _get_daily_pv_data(months)

    # Optional: merge daily weather features on 'date'
    if cfg.features.weather:
        weather_df = _get_daily_weather_data(months, cfg.features.weather, cfg.training.limit_training_data)
        df = pd.merge(df, weather_df, on="date", how="inner")
    else:
        weather_df = _get_daily_weather_data(months, ["cloud_cover"], cfg.training.limit_training_data)
        df = pd.merge(df, weather_df, on="date", how="inner")
        df = df.drop(columns=["cloud_cover"])

    # Optional: merge daily solar-position features on 'date'
    if cfg.features.solar_pos:
        sp_df = _get_daily_solar_pos_data(months, cfg.features.solar_pos)
        if sp_df is not None:
            df = pd.merge(df, sp_df, on="date", how="inner")

    # Optional: add date-based timestamp features (e.g., month, weekday, holidays)
    if cfg.features.timestamp:
        df = append_date_features(df)

    # If used as a feature: add daily clear-sky energy P_cs_kWh
    if cfg.training.use_clearsky_feature:
        csd = clearsky_daily(cfg, df["date"])
        df = pd.merge(df, csd, on="date", how="inner")

    # If normalization is enabled: filter tiny P_cs and create pv_norm = pv_yield / P_cs_kWh
    if cfg.training.normalize_by_clearsky:
        df = _normalize_dataframe(df, cfg.params.eps_pcs)

    return df


def _build_hourly_database(cfg: RunCfg,
                           months: int | List[int]
                           ) -> pd.DataFrame:
    """
   Assemble the hourly-resolution dataframe: PV → (optional) weather → (optional) solar-pos
    → (optional) timestamp features → (optional) clear-sky feature → (optional) normalization.

    :param cfg: Central configuration instance for the training run with all parameters
    :param months: Months to include
    :return: Hourly dataframe with requested features. If normalization is enabled, includes 'pv_norm'
    """
    # Base PV target at hourly resolution
    df = _get_hourly_pv_data(months)

    # Optional: merge hourly weather features on 'datetime'
    if cfg.features.weather:
        weather_df = _get_hourly_weather_data(months, cfg.features.weather, cfg.training.limit_training_data)
        df = pd.merge(df, weather_df, on="datetime", how="inner")
    else:
        weather_df = _get_hourly_weather_data(months, ["cloud_cover"], cfg.training.limit_training_data)
        df = pd.merge(df, weather_df, on="date", how="inner")
        df = df.drop(columns=["cloud_cover"])

    # Optional: merge hourly solar-position features on 'datetime'
    if cfg.features.solar_pos:
        sp_df = _get_hourly_solar_pos_data(months, cfg.features.solar_pos)
        df = pd.merge(df, sp_df, on="datetime", how="inner")

    # Optional: add time and date-based features.
    if cfg.features.timestamp:
        df = append_time_features(df)
        df = append_date_features(df, "datetime")

    # If used as a feature: add hourly clear-sky energy P_cs_kWh at the same timestamps
    if cfg.training.use_clearsky_feature:
        cs = clearsky_hourly(cfg, df["datetime"])
        df = pd.merge(df, cs, on="datetime", how="inner")

    # If normalization is enabled: filter tiny P_cs and create pv_norm = pv_yield / P_cs_kWh
    if cfg.training.normalize_by_clearsky:
        df = _normalize_dataframe(df, cfg.params.eps_pcs)

    return df


def _normalize_dataframe(df: pd.DataFrame, eps_pcs: float)-> pd.DataFrame:
    """
    Normalize PV yield by clear-sky energy with a small-denominator guard.

    Creates 'pv_norm = pv_yield / P_cs_kWh' after filtering rows where 'P_cs_kWh < EPS_PCS' to avoid division
    by near-zero values (night/low-irradiance hours).

    :param df: Input dataframe that must contain at least 'pv_yield' and 'P_cs_kWh'
    :param eps_pcs: Only normalize once this clear sky potential has been reached
    :return: Copy of the filtered dataframe with an added 'pv_norm' column
    """
    # Keep only rows with sufficiently large clear-sky energy to ensure stable ratios
    mask = df["P_cs_kWh"] >= eps_pcs
    df = df.loc[mask].copy()

    # Normalized target: unitless ratio of actual yield to clear-sky potential
    df["pv_norm"] = df["pv_yield"] / df["P_cs_kWh"]

    return df
