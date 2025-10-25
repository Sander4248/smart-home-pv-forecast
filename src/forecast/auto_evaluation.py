import calendar
import logging
from datetime import datetime, timedelta
from typing import List, Set, Tuple

from src.config.schema import RunCfg
from src.data.solar_power_dao import get_max_solar_timestamp
from src.data.weather_forecasts_dao import get_all_response_timestamps
from src.data.training_results_dao import get_session_id, get_results_data
from src.forecast.model_evaluation import evaluate_model

from src.forecast.model_evaluation import evaluate_baseline
from src.baselines.persistence import (
    baseline_y_minus_24h,
    baseline_mean_last7d_same_hour,
    baseline_same_dow_last4w_mean,
    baseline_daily_y_minus_1d,
    baseline_daily_mean_last7d,
    baseline_daily_same_dow_last4w_mean,
)

BASELINES_HOURLY = {
    "y-24h": baseline_y_minus_24h,
    "mean-7d": baseline_mean_last7d_same_hour,
    "same-dow-4w": baseline_same_dow_last4w_mean,
}

BASELINES_DAILY = {
    "y-1d": baseline_daily_y_minus_1d,
    "mean-7d": baseline_daily_mean_last7d,
    "same-dow-4w": baseline_daily_same_dow_last4w_mean,
}


def _determine_required_forecast_timestamps(min_ts: datetime,
                                            max_ts: datetime,
                                            months: List[int]
                                            ) -> List[datetime]:
    """
    Generates a complete list of potential forecast times within the year range [min_ts.year, max_ts.year]
    for the specified months and all hours (0–23).

    Important notes:
      - The function does NOT limit to the exact interval [min_ts, max_ts]; filtering occurs later
        (before appending the pairs).
      - Leap years are correctly handled by 'calendar.monthrange'.
      - Timestamps are naive 'datetime' objects (no time zone logic/DST).

    :param min_ts: Lower bound (datetime) for deriving the smallest year
    :param max_ts: Upper bound (datetime) for deriving the largest year
    :param months: Iterable of month numbers (1–12) to be considered
    :return: All generated forecast timestamps (per day and hour of the selected months)
    """
    forecast_timestamps = []
    for year in range(min_ts.year, max_ts.year + 1):
        for month in months:
            number_of_days = calendar.monthrange(year, month)[1]
            for day in range(1, number_of_days + 1):
                for hour in range(24):
                    forecast_timestamps.append(datetime(year, month, day, hour))

    return forecast_timestamps


def _determine_max_timestamp(forecast_horizon: int,
                             max_response_ts: datetime,
                             max_solar_data_ts: datetime
                             ) -> datetime:
    """
    Determines the maximum upper limit for evaluation times to avoid data gaps.

    Reason (Why?):
    Weather responses and actual PV values are not updated synchronously.
    Without this cap, evaluation times could lie outside the more recent data source:
    there would be responses without PV (or vice versa), empty DB selects, or even 'forecast_ts < response_ts'.
    The minimum of the two upper limits prevents these inconsistencies and ensures that data is available
    in both sources for each (response_ts, forecast_ts).

    Logic:
      - PV actual values must be available 'forecast_horizon' days later than the response time.
        Therefore, 'max_solar_data_ts' is reset by 'forecast_horizon' days.
      - The smaller of the two values (reset PV max timestamp, 'max_response_ts') is the safe upper limit.

    :param forecast_horizon: Horizon in days between response and forecast time
    :param max_response_ts: Maximum available response timestamp from the weather database
    :param max_solar_data_ts: Maximum available PV actual value timestamp
    :return: Safe maximum upper limit for generating and filtering timestamps
    """
    max_ts = min(max_solar_data_ts - timedelta(days=forecast_horizon), max_response_ts)

    return max_ts


def _get_response_time_bounds(available_response_timestamps: Set[datetime]) -> Tuple[datetime, datetime]:
    """
    Determines the minimum and maximum response times in the weather database.

    :return: min_response_ts, max_response_ts
    """
    min_response_ts = min(available_response_timestamps)
    max_response_ts = max(available_response_timestamps)

    return min_response_ts, max_response_ts


def _search_existing_response_timestamp(available_response_timestamps: Set[datetime],
                                        target_response_ts: datetime
                                        ) -> datetime:
    """
    Searches for an actual response timestamp based on the target response timestamp.

    If the response timestamp being searched for does not exist, the last response timestamp is used.
    The search is deliberately performed backwards and not for the closest one.

    The closest one could be younger than the one being searched for. -> This can result in the forecast timestamp
    being older than the response timestamp, which is not possible and leads to no result in the select, which in
    turn leads to an error in the program. This can occur if several response timestamps are missing in the
    weather forecast.

    :param available_response_timestamps: Set with available response timestamps in the database
    :param target_response_ts: Current response timestamp being searched for
    :return: Available response timestamp that is exactly the one you are looking for or an older available one
    """
    found_not_exact_ts = True
    while found_not_exact_ts:

        # Determine existing response timestamp
        if target_response_ts in available_response_timestamps:
            # Exact timestamp available
            response_ts = target_response_ts
            found_not_exact_ts = False
        else:
            # Search response timestamp one hour earlier in next round
            target_response_ts -= timedelta(hours=1)

    return response_ts


def run_auto_evaluate(cfg: RunCfg) -> None:
    """
    Evaluates all trained models from the last active session and evaluates the corresponding baselines.

    Procedure:
      1) Determines the last 'session_id'.
      2) Loads all model results (result_id + training months) from this session.
      3) Determines min/max response timestamps for the weather data and all actual response timestamps.
      4) Determines the maximum permissible timestamp (intersection of weather and PV data to avoid inconsistencies).
      5) Generates all candidate forecast times (per training month and hour) for each model.
         For each forecast time point, the matching/earlier response time point is searched for.
      6) Calls 'evaluate_model(...)' with the found (response_ts, forecast_ts) pairs.
      7) Calls 'evaluate_baseline' for each baseline evaluation.

    Difference 'daily' data resolution:
      - The prediction data from a response is used.
      - The target request is at 12 noon.
      - The 24 data records are aggregated.

    Assumptions:
      - 'data_resolution' corresponds to the table prefix in the result database (e.g., "<res>_models").
      - Date/time strings are in the format “%Y-%m-%d %H:%M:%S” (naive datetimes).
      - If no exact response timestamp exists, the last older one is selected backwards.

    :param cfg: Central configuration instance for the training run with all parameters
    :return: None (Side effects: DB read accesses, model evaluation call).
    """

    logging.info("Start evaluation of Model.")

    available_response_timestamps = get_all_response_timestamps()

    # Determine the minimum and maximum timestamp to narrow down the determination of the evaluation timestamps.
    min_response_ts, max_response_ts = _get_response_time_bounds(available_response_timestamps)
    max_solar_data_ts = get_max_solar_timestamp(cfg.training.data_resolution)
    max_ts = _determine_max_timestamp(cfg.evaluation.forecast_horizon, max_response_ts, max_solar_data_ts)

    # Evaluate each model from the last session
    for result in get_results_data(cfg.training.data_resolution, get_session_id()):
        result_id = result[0]
        try:
            months = [int(x) for x in result[1].split(";")]
        except AttributeError:
            months = [int(result[1])]
        eval_timestamps = []

        for forecast_ts in _determine_required_forecast_timestamps(min_response_ts, max_ts, months):

            # If a response timestamp exists, it is assumed that all forecast timestamps exist
            target_response_ts = forecast_ts - timedelta(days=cfg.evaluation.forecast_horizon)
            if cfg.training.data_resolution == "daily":
                target_response_ts = target_response_ts.replace(hour=12)

            # Check whether the timestamp you are looking for is actually between the minimum and maximum timestamps
            if min_response_ts <= target_response_ts <= max_ts:
                response_ts = _search_existing_response_timestamp(available_response_timestamps,
                                                                  target_response_ts)
                eval_timestamps.append((response_ts, forecast_ts))

        # Evaluate model if weather forecast and solar data records are available for it
        if eval_timestamps:
            evaluate_model(cfg, result_id, timestamps=eval_timestamps)

        # Evaluate baseline
        if cfg.training.data_resolution == "hourly":
            baselines = BASELINES_HOURLY
        else:
            baselines = BASELINES_DAILY

        for name, fn in baselines.items():
            evaluate_baseline(cfg, result_id, name, fn, timestamps=eval_timestamps)

    logging.info("Evaluation complete.")