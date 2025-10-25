from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from math import isinf
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config.logging_config import setup_logging
from src.config.schema import RunCfg
from src.data.solar_power_dao import get_hourly_yield_by_timestamp, get_daily_yield_by_timestamp
from src.data.training_results_dao import load_model_from_db, save_evaluation_results
from src.data.weather_forecasts_dao import get_min_forecast_ts
from src.forecast.yield_prediction import predict_hourly_yield, predict_daily_yield
from src.utils import create_comma_separated_string


def _calculate_forecast_error(actual_yield: float,
                              forecast_yield: float
                              ) -> Tuple[float, float]:
    """
    Calculates the forecast error between actual and predicted PV yield.

    Rules for the relative error:
    - If both actual and forecast yields are 0, the relative error is defined as 0.
    - If the actual yield is 0 (but the forecast is not), the relative error is set to infinity.
    - Otherwise, the relative error is calculated as a percentage of the actual yield.

    In addition, the signed difference between actual and forecast yield is returned.

    :param actual_yield: The measured PV yield [kWh].
    :param forecast_yield: The predicted PV yield [kWh].
    :return: Tuple of (difference in kWh, relative error in %).
    """
    if forecast_yield == 0 and actual_yield == 0:
        rel_error = 0.0
    elif actual_yield == 0:
        rel_error = float("inf")
    else:
        rel_error = (forecast_yield - actual_yield) / actual_yield * 100

    difference = round(actual_yield - forecast_yield, 1)

    return difference, rel_error


def _calculate_smape(y_true: np.ndarray | List[float],
                     y_pred: np.ndarray | List[float]
                     ) -> float:
    """
    Computes Symmetric Mean Absolute Percentage Error (sMAPE) in percent.

    sMAPE = mean( |y_pred - y_true| / ((|y_true| + |y_pred|) / 2) ) * 100

    Notes:
    - For purely nocturnal hours where both true and predicted are zero, the contribution is treated as 0
      by replacing a zero denominator with 1.0 (has no impact because numerator is also 0).
    :param y_true: Ground-truth values (array-like).
    :param y_pred: Predicted values (array-like).
    :return: sMAPE in percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    # For pure night-time 0–0 hours -> contribution 0
    denom_safe = np.where(denom == 0, 1.0, denom)

    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom_safe)


def _calculate_mape_pos(y_true: np.ndarray | List[float],
                        y_pred: np.ndarray | List[float]
                        ) -> float:
    """
    Computes MAPE (Mean Absolute Percentage Error) in percent, but only over strictly positive y_true.

    This avoids division by zero and prevents night-time zeros from dominating the metric.

    :param y_true: Ground-truth values (array-like).
    :param y_pred: Predicted values (array-like).
    :return: MAPE in percent for samples with y_true > 0. Returns NaN if no positive samples exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0

    if not np.any(mask):
        return np.nan

    return 100.0 * np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))


def _calculate_nrmse_range(y_true: np.ndarray | List[float],
                           y_pred: np.ndarray | List[float]
                           ) -> float:
    """
    Computes the range-normalized RMSE: RMSE / (max(y_true) - min(y_true)).

    :param y_true: Ground-truth values (array-like).
    :param y_pred: Predicted values (array-like).
    :return: nRMSE in the unit of [relative to range]. Returns NaN if range is zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    rng = y_true.max() - y_true.min()

    return rmse / rng if rng > 0 else np.nan


def _calculate_energy_balance_error(y_true: np.ndarray | List[float],
                                    y_pred: np.ndarray | List[float]
                                    ) -> float:
    """
    Computes the energy balance error over the whole horizon: sum(pred) - sum(true).

    Interpretation:
    - Positive value -> model overestimates total energy.
    - Negative value -> model underestimates total energy.

    :param y_true: Ground-truth values (array-like).
    :param y_pred: Predicted values (array-like).
    :return: Energy balance error in kWh.
    """
    return float(np.sum(y_pred) - np.sum(y_true))


def _calculate_tolerance_buckets(rel_errors: List[float]) -> Dict[str, int]:
    """
    Buckets relative errors (in %) into tolerance classes.

    Buckets: "≤25%", "≤50%", "≤75%", "≤100%", ">100%", "inf"

    Notes:
    - Infinite errors (actual = 0, forecast > 0) are counted in the "inf" bucket.
    - All finite errors are bucketed by their absolute value.

    :param rel_errors: List of relative errors (in %) where +inf may occur.
    :return: Dict mapping bucket labels to counts.
    """
    buckets = {"≤25%": 0, "≤50%": 0, "≤75%": 0, "≤100%": 0, ">100%": 0, "inf": 0}
    for v in rel_errors:
        if isinf(v):
            buckets["inf"] += 1
        else:
            av = abs(v)
            if av <= 25:
                buckets["≤25%"] += 1
            elif av <= 50:
                buckets["≤50%"] += 1
            elif av <= 75:
                buckets["≤75%"] += 1
            elif av <= 100:
                buckets["≤100%"] += 1
            else:
                buckets[">100%"] += 1

    return buckets


def _hourly_peak_timing(df_hours: pd.DataFrame) -> Dict[str, float | int]:
    """
    Computes per-day peak-timing statistics from hourly data.

    Compute, for each calendar day, the time difference in hours between the predicted daily peak (argmax of y_pred)
    and the true daily peak (argmax of y_true). If multiple hours share the same maximum, the earliest is chosen.

    :param df_hours: Hourly data with timestamps and true/predicted yields.
    :return: Dict with:
             - count_days: number of days with valid peak comparison
             - median_h: median difference [h] (predicted - true)
             - p90_h: 90th percentile of differences [h]
             - min_h: minimum difference [h]
             - max_h: maximum difference [h]
             Values are NaN if no day had valid peaks.
    """
    # Derive calendar date per row for grouping
    df_hours["date"] = df_hours["timestamp"].dt.date

    diffs: List[float] = []
    for _, g in df_hours.groupby("date"):
        # Work on a copy to avoid chained assignment surprises
        g = g.copy()

        # Peak times (if ties: take the earliest by idxmax behavior)
        t_true = g.loc[g["y_true"].idxmax(), "timestamp"] if len(g) else None
        t_pred = g.loc[g["y_pred"].idxmax(), "timestamp"] if len(g) else None

        if t_true is not None and t_pred is not None:
            delta_h = (t_pred - t_true).total_seconds() / 3600.0
            diffs.append(delta_h)

    if not diffs:
        return {"count_days": 0, "median_h": np.nan, "p90_h": np.nan, "min_h": np.nan, "max_h": np.nan}

    array_diffs = np.array(diffs)

    return {
        "count_days": int(len(array_diffs)),
        "median_h": float(np.median(array_diffs)),
        "p90_h": float(np.percentile(array_diffs, 90)),
        "min_h": float(array_diffs.min()),
        "max_h": float(array_diffs.max()),
    }


def prepare_and_save_evaluation_results(cfg: RunCfg,
                                        data: dict,
                                        result_id: int,
                                        start: datetime | None,
                                        end: datetime | None,
                                        months: str | None,
                                        model_label: str | None = None
                                        ) -> None:
    """
    Saves the evaluation results in the database.

    - Maps the logging dictionary (with nested dictionaries) to a new dictionary where the
      keys correspond to the column names of the database table.
    - Columns and placeholders for the insert query are derived from the mapping dictionary.

    :param cfg: Central configuration instance for the training run with all parameters
    :param data: Dictionary with data to store (nested dictionary)
    :param result_id: Identifier of the trained model to use inside 'predict_yield'.
    :param start: Optional. Start date (00:00 of this day is the first hour evaluated).
    :param end: Optional. End date (23:00 of this day is the last hour evaluated).
    :param months: Optional. CSV with evaluated month
    :param model_label: Label for model or baseline used
    :return: Nothing, but stores evaluation results in the database.
    """

    if not model_label:
        model_label = cfg.model.type

    db_data = {
        "result_id": result_id if result_id is not None else -1,
        "start": None,
        "end": None,
        "months": None,
        "night_clamp": cfg.evaluation.night_clamp,
        "night_clamp_threshold": cfg.evaluation.night_clamp_threshold,
        "forecast_horizon": cfg.evaluation.forecast_horizon if cfg.evaluation.forecast_horizon is not None else -1,
        "mae": data["metrics"]["MAE_kWh"],
        "rmse": data["metrics"]["RMSE_kWh"],
        "nrmse": data["metrics"]["nRMSE_range"],
        "bias_me": data["metrics"]["Bias_ME_kWh"],
        "r2": data["metrics"]["R2"],
        "smape": data["metrics"]["sMAPE_%"],
        "mape_y": data["metrics"]["MAPE_%_y>0"],
        "forecast_sum": data["sums"]["forecast_sum_kWh"],
        "actual_sum": data["sums"]["actual_sum_kWh"],
        "energy_balance_error": data["sums"]["energy_balance_error_kWh"],
        "lt_25": data["tolerance_buckets"]["≤25%"],
        "lt_50": data["tolerance_buckets"]["≤50%"],
        "lt_75": data["tolerance_buckets"]["≤75%"],
        "lt_100": data["tolerance_buckets"]["≤100%"],
        "gt_100": data["tolerance_buckets"][">100%"],
        "inf": data["tolerance_buckets"]["inf"],
        "model_label": model_label
    }

    if cfg.training.data_resolution == "hourly":
        db_data.update({
            "n_hours": data["counts"]["n_hours"],
            "n_daylight_hours": data["counts"]["n_daylight_hours"],
            "n_inf": data["counts"]["n_inf"],
            "count_days": data["peak_timing_hours"]["count_days"],
            "median_h": data["peak_timing_hours"]["median_h"],
            "p90_h": data["peak_timing_hours"]["p90_h"],
            "min_h": data["peak_timing_hours"]["min_h"],
            "max_h": data["peak_timing_hours"]["max_h"]
        })

    # Round floats in db_data to 1 decimal place
    db_data = {k: round(v, 2) if isinstance(v, float) else v for k, v in db_data.items()}

    if months:
        db_data["months"] = months
    else:
        db_data["start"] = start.isoformat()
        db_data["end"] = end.isoformat()

    save_evaluation_results(cfg.training.data_resolution, db_data)

    logging.info(f"Evaluation successfully stored in DB. Resolution: {cfg.training.data_resolution}, "
                 f"Label: {model_label}")


def _evaluate_hourly_pv_forecasts(cfg: RunCfg,
                                  result_id: int,
                                  timestamps: List[Tuple[datetime, datetime]] = None,
                                  start: datetime = None,
                                  end: datetime = None
                                  ) -> Tuple[List[datetime], List[float], List[float], List[float]]:
    """
    Evaluates hourly PV forecasts against actual yields and collects the results.

    Process:
     - If ‘timestamps’ are specified: For each pair of (response_ts, forecast_ts), the forecast is calculated
       and compared with the actual values.
     - If no ‘timestamps’ are specified: Iterates over the period from 'start' to 'end' and calculates hourly
       forecasts for each hour of the day. (If timestamps and start and end dates are passed, the dates are discarded.)
     - The results (timestamps, forecasts, actual values, relative errors) are collected in lists and returned.

    :param cfg: Central configuration instance for the training run with all parameters
    :param result_id: Identifier of the trained model to use inside 'predict_yield'.
    :param timestamps: Optional. List of tuples with response and forecast timestamps.
    :param start: Optional. Start date (00:00 of this day is the first hour evaluated).
    :param end: Optional. End date (23:00 of this day is the last hour evaluated).
    :return: Updated Lists with results
    """
    forecast_timestamps: List[datetime] = []  # Collection of all forecast timestamps
    y_pred: List[float] = []  # Collection of predicted values
    y_true: List[float] = []  # Collection of actual values
    rels: List[float] = []  # Collection of relative error values in %

    model = load_model_from_db(cfg.training.data_resolution, result_id)

    # Evaluate passed timestamps
    if timestamps:
        for response_ts, forecast_ts in timestamps:
            forecast_yield, actual_yield, apparent_elevation = (
                predict_hourly_yield(cfg, forecast_ts, model=model, response_ts=response_ts))

            _update_pv_forecast_metrics(actual_yield, forecast_yield, cfg.evaluation.night_clamp,
                                        cfg.evaluation.night_clamp_threshold, forecast_ts, forecast_timestamps,
                                        y_pred, y_true, rels, apparent_elevation, cfg=cfg)

    # Evaluate the specified period (timestamps are generated for this purpose)
    else:
        # Minimum timestamp for which forecast data is available
        min_ts = get_min_forecast_ts() + timedelta(hours=cfg.evaluation.forecast_horizon * 24)

        current = start
        # Evaluate all days between start and end date
        while current <= end:
            # Evaluate all 24 hours of the current day
            for h in range(24):
                forecast_ts = current + timedelta(hours=h)

                # If the current timestamp is less than the minimum timestamp, proceed to the next timestamp
                if forecast_ts < min_ts:
                    continue

                forecast_yield, actual_yield, apparent_elevation = (
                    predict_hourly_yield(cfg, forecast_ts, model=model))

                _update_pv_forecast_metrics(actual_yield, forecast_yield, cfg.evaluation.night_clamp,
                                            cfg.evaluation.night_clamp_threshold, forecast_ts, forecast_timestamps,
                                            y_pred, y_true, rels, apparent_elevation, cfg=cfg)

            current += timedelta(days=1)

    return forecast_timestamps, y_pred, y_true, rels


def _evaluate_daily_pv_forecasts(cfg: RunCfg,
                                 result_id: int,
                                 timestamps: List[Tuple[datetime, datetime]] = None,
                                 start: datetime = None,
                                 end: datetime = None
                                 ) -> Tuple[List[datetime], List[float], List[float], List[float]]:
    """
    Evaluates daily PV forecasts against actual yields and collects the results.

    Process:
     - If ‘timestamps’ are specified: For each pair of (response_ts, forecast_ts), the forecast is calculated
       and compared with the actual values.
     - If no ‘timestamps’ are specified: Not implemented
     - The results (timestamps, forecasts, actual values, relative errors) are collected in lists and returned.

    :param cfg: Central configuration instance for the training run with all parameters
    :param result_id: Identifier of the trained model to use inside 'predict_yield'.
    :param timestamps: Optional. List of tuples with response and forecast timestamps.
    :param start: Optional. Start date (00:00 of this day is the first hour evaluated).
    :param end: Optional. End date (23:00 of this day is the last hour evaluated).
    :return: Updated Lists with results
    """
    forecast_timestamps: List[datetime] = []  # Collection of all forecast timestamps
    y_pred: List[float] = []  # Collection of predicted values
    y_true: List[float] = []  # Collection of actual values
    rels: List[float] = []  # Collection of relative error values in %

    model = load_model_from_db(cfg.training.data_resolution, result_id)

    # Evaluate passed timestamps
    if timestamps:

        days = _group_by_forecast_date(timestamps)
        for day in days:
            if len(day) != 24:
                logging.error(f"Response day '{day[0][0].date()}' does not have 24 hour forecasts")
            forecast_yield, actual_yield = predict_daily_yield(cfg, daily_timestamps=day, model=model)

            _update_pv_forecast_metrics(actual_yield, forecast_yield, cfg.evaluation.night_clamp,
                                        cfg.evaluation.night_clamp_threshold, day[0][1].date(),
                                        forecast_timestamps, y_pred, y_true, rels, cfg=cfg)

    # Evaluate the specified period (timestamps are generated for this purpose)
    else:
        logging.critical("Daily evaluation with 'start' and 'end' dates not yet implemented.'")
        exit(1)

    return forecast_timestamps, y_pred, y_true, rels


def _group_by_forecast_date(data: List[Tuple[datetime, datetime]]) -> List[List[Tuple[datetime, datetime]]]:
    """
    Groups a list of response/forecast tuples by forecast date.

    The expected input is a list whose elements are each a tuple of (response, forecast).

    Process:
    - A dictionary is created that stores a list for each unique date
    - Each (response, forecast) tuple is assigned based on the date of forecast
    - Finally, a list of all groups is returned, sorted by date

    :param data: List of (response, forecast) tuples
    :return: List of lists, where each inner list contains all (response, forecast) tuples for a given forecast
             date, sorted chronologically.
    """
    groups = defaultdict(list)
    for response, forecast in data:
        groups[forecast.date()].append((response, forecast))

    return [groups[day] for day in sorted(groups)]


def _update_pv_forecast_metrics(actual_yield: float,
                                forecast_yield: float,
                                night_clamp: bool,
                                night_clamp_threshold: float,
                                forecast_timestamp: datetime,
                                forecast_timestamps: List[datetime],
                                y_pred: List[float],
                                y_true: List[float],
                                rels: List[float],
                                apparent_elevation: float = None,
                                baseline: bool = False,
                                cfg = None
                                ) -> None:
    """
    Processes a PV yield forecast, calculates the deviation from the actual PV yield and updates the passed results and
    time series lists.

    Night clamp is only useful for evaluating hourly models and should be deactivated for daily models.
    Night clamp is deactivated for baseline evaluations (regardless of data resolution).

    With night clamp, the pv yield prediction is automatically set to 0 if it falls below a certain threshold value.
    This threshold value is set via 'night_clamp_threshold'.
    A reasonable value here would be between 0 and 10 degrees.

    Process:
     - Calculates the absolute and relative deviation between actual and predicted yield.
     - Logs the forecast, actual, and error values.
     - If night_clamp=True, If true, forecast yields are set to 0 if the apparent elevation is below
       the threshold 'night_clamp_threshold'.
     - Writes the results to the transferred lists

    :param actual_yield: Measured actual PV yield in kWh
    :param forecast_yield: Forecast PV yield in kWh
    :param night_clamp: Use night_clamp True/False
    :param night_clamp_threshold: Threshold value, above which a value is treated as night
    :param forecast_timestamp: Timestamp of the current forecast
    :param forecast_timestamps: Collection of all forecast timestamps
    :param y_pred: Collection of predicted values
    :param y_true: Collection of actual values
    :param rels: Collection of relative error values in %
    :param baseline: Default = False. If true, this is a baseline evaluation and night clamp is disabled.
    :return: Nothing, but updates the lists that were passed to it.
    """
    # Compute signed difference (actual - forecast) and relative error
    difference, rel_error = _calculate_forecast_error(actual_yield, forecast_yield)

    # Optionally: Night clamp. if apparent elevation is below threshold, then forecast_yield is automatically 0
    if not baseline and night_clamp:
        if apparent_elevation is not None:
            if apparent_elevation < night_clamp_threshold:
                forecast_yield = 0.0
                difference, rel_error = _calculate_forecast_error(actual_yield, forecast_yield)
        else:
            logging.warning("Night clamp could not be applied because apparent elevation is None.")

    if not baseline:
        logging.debug(
            f"PV yield {forecast_timestamp} - Forecast: {forecast_yield:.1f} kWh - "
            f"Actual: {actual_yield:.1f} kWh - Dif: {difference:.1f} kWh - "
            f"Relative Err: {rel_error:.1f} %"
        )

    # Accumulate series
    forecast_timestamps.append(forecast_timestamp)
    y_pred.append(float(forecast_yield))
    y_true.append(float(actual_yield))
    rels.append(rel_error)


def _determine_and_collect_metrics(data_resolution: str,
                                   forecast_timestamps: List[datetime],
                                   rels: List[float],
                                   y_pred: List[float],
                                   y_true: List[float]
                                   ) -> dict:
    """
    Calculate and collect evaluation metrics for forecast performance.

    This function computes a set of statistical and error-based metrics comparing predicted values ('y_pred')
    against true values ('y_true'). Results are returned as a structured dictionary, including metrics,
    aggregated sums, counts, tolerance buckets, and (for hourly data) peak timing statistics.

    :param data_resolution: Resolution of the data.
           If "hourly", additional daily peak timing statistics and counts are calculated
    :param forecast_timestamps: Sequence of timestamps corresponding to each forecasted value.
    :param rels: Relative errors per time step, used for tolerance bucket calculation.
    :param y_pred: Predicted values of the target variable.
    :param y_true: Actual (observed) values of the target variable
    :return: A dictionary with the following keys:
             - "metrics": dict of error metrics (MAE, RMSE, nRMSE, bias, R², sMAPE, MAPE).
             - "sums": dict with total forecast and actual energy sums and energy balance error.
             - "counts": dict with counts of hours, daylight hours, and infinities (only for hourly data).
             - "tolerance_buckets": dict of tolerance bucket statistics from relative errors.
             - "peak_timing_hours": dict with daily peak timing statistics (only for hourly data).
    """
    # Convert to arrays for vectorized metrics
    y_pred_arr = np.array(y_pred, dtype=float)
    y_true_arr = np.array(y_true, dtype=float)

    # Basic metrics (all hours)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    nrmse = float(_calculate_nrmse_range(y_true_arr, y_pred_arr))
    me_bias = float(np.mean(y_true_arr - y_pred_arr))  # positive => model underestimates on average
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    ebe = _calculate_energy_balance_error(y_true_arr, y_pred_arr)

    # Robust percentage metrics
    smape_val = float(_calculate_smape(y_true_arr, y_pred_arr))
    mape_pos_val = float(_calculate_mape_pos(y_true_arr, y_pred_arr))

    # Store basic metrics in a dictionary
    metrics = {
        "MAE_kWh": mae,
        "RMSE_kWh": rmse,
        "nRMSE_range": nrmse,
        "Bias_ME_kWh": me_bias,
        "R2": r2,
        "sMAPE_%": smape_val,
        "MAPE_%_y>0": mape_pos_val,
    }

    # Tolerance buckets based on hour-level relative errors
    buckets = _calculate_tolerance_buckets(rels)

    # Build a DataFrame for downstream analyses/plots (e.g., diurnal profiles)
    df_hours = pd.DataFrame({
        "timestamp": pd.to_datetime(forecast_timestamps),
        "y_true": y_true_arr,
        "y_pred": y_pred_arr
    })

    # Daily peak timing statistics and counts
    if data_resolution == "hourly":
        # Daily peak timing statistics
        peak_stats = _hourly_peak_timing(df_hours)
        counts = {
            "n_hours": int(len(y_true_arr)),
            "n_daylight_hours": int((y_true_arr > 0).sum()),
            "n_inf": int(sum(1 for r in rels if isinf(r))),
        }
    else:
        peak_stats = {}
        counts = {}

    # Totals
    sums = {
        "forecast_sum_kWh": float(np.sum(y_pred_arr)),
        "actual_sum_kWh": float(np.sum(y_true_arr)),
        "energy_balance_error_kWh": float(ebe),
    }

    # Collect dicts in a parent dict
    log_data = {
        "metrics": metrics,
        "sums": sums,
        "counts": counts,
        "tolerance_buckets": buckets,
        "peak_timing_hours": peak_stats,
    }
    return log_data


def evaluate_model(cfg: RunCfg,
                   result_id: int,
                   timestamps: List[Tuple[datetime, datetime]] = None,
                   start: datetime = None,
                   end: datetime = None,
                   ) -> None:
    """
    Computes a set of robust metrics plus daily peak-timing statistics for a given period of time.

    Results are logged; optionally they can be stored in a database (not implemented here).

    :param cfg: Central configuration instance for the training run with all parameters
    :param result_id: Identifier of the trained model to use inside 'predict_yield'.
    :param timestamps: Optional. List of tuples with response and forecast timestamps.
    :param start: Optional. Start date (00:00 of this day is the first hour evaluated).
    :param end: Optional. End date (23:00 of this day is the last hour evaluated).
    :return: Nothing.
    """

    # Check whether conditions for model evaluation are met
    if timestamps is None and (start is None or end is None):
        logging.critical("Model cannot be evaluated. No timestamps or start and end dates available. "
                         "Evaluation is terminated.")
        exit(1)

    # Get lists of pre-evaluated results
    if cfg.training.data_resolution == "hourly":
        forecast_timestamps, y_pred, y_true, rels = (
            _evaluate_hourly_pv_forecasts(cfg, result_id, timestamps, start, end))
    else:  # daily
        forecast_timestamps, y_pred, y_true, rels = (
            _evaluate_daily_pv_forecasts(cfg, result_id, timestamps, start, end))

    if cfg.evaluation.aggregate_hourly_to_daily:
        forecast_timestamps, rels, y_pred, y_true = (
            _aggregate_hourly_to_daily(cfg, forecast_timestamps, rels, y_pred, y_true))

    # Evaluate the collected pre-evaluated results
    log_data = _determine_and_collect_metrics(cfg.training.data_resolution, forecast_timestamps, rels, y_pred, y_true)

    # Print info from current evaluation
    if timestamps:
        months = {ts[1].month for ts in timestamps}
        months_csv = create_comma_separated_string(sorted(months))
        logging.info(f"Result ID: {result_id} | Months: {months_csv} | "
                     f"Forecast-horizon: {cfg.evaluation.forecast_horizon}")
    else:
        months_csv = None
        logging.info(f"Result ID: {result_id} | {start.date()} - {end.date()} | "
                     f"Forecast-horizon: {cfg.evaluation.forecast_horizon}")

    # Print metrics
    for category, nested_dict in log_data.items():

        # Print headline only if data available for current category
        if nested_dict:
            logging.info(f">>> {category.upper()} <<<")

        # Print all metrics from current category
        for key, value in nested_dict.items():
            if isinstance(value, float):
                value = round(value, 2)
            logging.info(f"  {key}: {value}")

    # Prepare evaluation results and save them in the database.
    if cfg.evaluation.store:
        prepare_and_save_evaluation_results(cfg, log_data, result_id, start, end, months_csv)


def _aggregate_hourly_to_daily(cfg: RunCfg,
                               forecast_timestamps: List[datetime],
                               rels: List[float],
                               y_pred: List[float],
                               y_true: List[float]):
    """
    EXPERIMENTAL: Aggregate hourly forecasts/targets into daily totals and recompute a per-day error metric.

    Sets 'cfg.training.data_resolution = "daily"' → This leads to inconsistencies in the database.
    Training is logged in the ‘hourly’ tables and evaluation in the ‘daily_evaluate’ table.

    Notes
    -----
    - This implementation drops a hard-coded date (2025-07-13). Consider moving this into
      'cfg.training.daily_aggregation.exclude_dates' for flexibility.
    - Grouping is performed on the *date* (no timezone info). Ensure your timestamps are already
      in the correct timezone or are timezone-naive but consistent.
      
    :param cfg: Central configuration instance for the training run with all parameters
    :param forecast_timestamps: Hourly timestamps aligned with 'y_pred' and 'y_true'
    :param rels: Hour-level error values (unused for the aggregation itself; kept for API parity)
    :param y_pred: Hourly predictions.
    :param y_true: Hourly ground truth values.
    :return: forecast_timestamps, rels, y_pred, y_true
    """
    # Assemble a single DataFrame for easy grouping and aggregation.
    df = pd.DataFrame({
        "forecast_timestamps": forecast_timestamps,
        "y_pred": y_pred,
        "y_true": y_true,
        "rels": rels
    })

    # Ensure timestamps are parsed as pandas datetimes.
    df["forecast_timestamps"] = pd.to_datetime(df["forecast_timestamps"])

    # Remove date → first day with weather forecast data is complete
    date_to_remove = pd.to_datetime("2025-07-13").date()
    df = df[df["forecast_timestamps"].dt.date != date_to_remove]

    # Group by calendar day (drop time component) and aggregate to daily level.
    # Current behavior: SUM over hours -> daily totals.
    daily_sum = df.groupby(df["forecast_timestamps"].dt.date)[["y_pred", "y_true"]].sum().reset_index()

    # Recompute the daily error metric using your existing helper.
    daily_sum["rels"] = daily_sum.apply(
        lambda row: _calculate_forecast_error(row["y_true"], row["y_pred"])[1],
        axis=1
    )

    # Convert columns back to lists for the existing return contract.
    forecast_timestamps = daily_sum["forecast_timestamps"].tolist()
    y_pred = daily_sum["y_pred"].tolist()
    y_true = daily_sum["y_true"].tolist()
    rels = daily_sum["rels"].tolist()

    # Record that downstream components are now working at daily resolution.
    cfg.training.data_resolution = "daily"

    return forecast_timestamps, rels, y_pred, y_true


def evaluate_baseline(cfg: RunCfg,
                      result_id: int,
                      baseline_name: str,
                      baseline_fn,  # Callable[[datetime], float|None]
                      timestamps: List[Tuple[datetime, datetime]],
                      start: datetime = None,
                      end: datetime = None,
                      ) -> None:
    """
    Evaluate a baseline forecast model (e.g. persistence baseline) in the same style as 'evaluate_model(...)'.

    This function generates forecasts using a given baseline function, compares them with actual PV yield data,
    calculates error metrics, and optionally stores the results in the evaluation database.

    Process
      1. Collect time series of predictions and actuals for the given resolution ("hourly" or "daily")
      2. Compute forecast error metrics, tolerance buckets, and peak statistics via '_determine_and_collect_metrics()'
      3. Optionally persist the results to the evaluation database with a  model label of the form 'baseline:<name>'

    Notes
      - For daily resolution, the function aggregates forecasts at the day level and uses midnight timestamps
      - If no valid points are available ('y_true' empty), a warning is logged and evaluation is skipped

    :param cfg: Central configuration instance for the training run with all parameters
    :param result_id: Identifier of the forecast run or experiment
    :param baseline_name: Human-readable name of the baseline (used in DB persistence label)
    :param baseline_fn: Function returning the baseline forecast value for a given timestamp
    :param timestamps: List of (response_time, forecast_time) pairs, defining when forecasts
                       were issued and for which target timestamps
    :param start: Evaluation start date for filtering persistence.
    :param end: Evaluation end date for filtering persistence.
    :return: None: The function does not return anything, but may log warnings and  persist results to the database.
    """

    forecast_timestamps: List[datetime] = []
    y_pred: List[float] = []
    y_true: List[float] = []
    rels: List[float] = []

    if cfg.training.data_resolution == "hourly":

        # Get the forecast value and actual value for each hour
        for response_ts, forecast_ts in timestamps:

            # Retrieve actual pv yield and prediction value from baseline function
            actual = get_hourly_yield_by_timestamp(
                forecast_ts.strftime("%Y-%m-%d"),
                forecast_ts.strftime("%H:%M:%S"),
            )
            pred = baseline_fn(forecast_ts)

            # If one of the values is missing → skip
            if pred is None or actual is None:
                continue

            _update_pv_forecast_metrics(
                actual, float(pred), cfg.evaluation.night_clamp, cfg.evaluation.night_clamp_threshold,
                forecast_ts, forecast_timestamps, y_pred, y_true, rels, baseline=True, cfg=cfg)
    else:  # daily

        # Grouping: each day should only occur once
        grouped = {}

        for _, forecast_ts in timestamps:
            key = forecast_ts.date()
            grouped.setdefault(key, forecast_ts)

        for day, forecast_ts in grouped.items():

            # Retrieve actual value for the day
            actual = get_daily_yield_by_timestamp(forecast_ts.strftime("%Y-%m-%d"))

            # Retrieve prediction value from baseline function
            pred = baseline_fn(forecast_ts)
            if pred is None or actual is None:
                continue

            # For daily values, 00:00 is used as the timestamp.
            ts_midnight = datetime(forecast_ts.year, forecast_ts.month, forecast_ts.day)

            _update_pv_forecast_metrics(
                actual, float(pred), cfg.evaluation.night_clamp, cfg.evaluation.night_clamp_threshold,
                ts_midnight, forecast_timestamps, y_pred, y_true, rels, baseline=True, cfg=cfg)

    # If no valid data was collected -> Warning and termination
    if not y_true:
        logging.warning(f"[evaluate_baseline] '{baseline_name}': No evaluable points.")
        return

    # Calculation of metrics
    log_data = _determine_and_collect_metrics(cfg.training.data_resolution, forecast_timestamps, rels, y_pred, y_true)

    # Extract months from the forecast timestamps (for later DB storage)
    if timestamps:
        months = {ts[1].month for ts in timestamps}
        months_csv = create_comma_separated_string(sorted(months))
    else:
        months_csv = None

    # Prepare evaluation results and save them in the database.
    if cfg.evaluation.store:
        prepare_and_save_evaluation_results(cfg, log_data, result_id=result_id, start=start, end=end,
                                            months=months_csv, model_label=f"baseline: {baseline_name}",
        )
    return


if __name__ == "__main__":
    setup_logging()
    cfg = RunCfg()
    result_id = 447
    start = datetime(2025, 7, 1)
    end = datetime(2025, 7, 31)

    evaluate_model(cfg, result_id, start=start, end=end)
