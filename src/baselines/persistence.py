
from __future__ import annotations
from datetime import datetime, timedelta

from src.data.solar_power_dao import get_hourly_yield_by_timestamp, get_daily_yield_by_timestamp


# ----------------------------
# Hourly baselines
# ----------------------------
def baseline_y_minus_24h(ts: datetime) -> float | None:
    """
    Takes the value of the same time 24 hours ago as the baseline.

    :param ts: Timestamp of the forecast
    :return: PV yield (kWh) from 24 hours ago or None
    """
    y = ts - timedelta(hours=24)

    return get_hourly_yield_by_timestamp(y.strftime("%Y-%m-%d"), y.strftime("%H:%M:%S"))


def baseline_mean_last7d_same_hour(ts: datetime) -> float | None:
    """
    Average value of the last 7 days at the same hour as baseline.

    :param ts: Timestamp of the forecast
    :return: Average PV yield (kWh) of the last 7 days at the same time or None
    """
    vals: list[float] = []
    for d in range(1, 8):
        y = ts - timedelta(days=d)
        v = get_hourly_yield_by_timestamp(y.strftime("%Y-%m-%d"), y.strftime("%H:%M:%S"))
        if v is not None:
            vals.append(v)

    return float(sum(vals) / len(vals)) if vals else None


def baseline_same_dow_last4w_mean(ts: datetime) -> float | None:
    """
    Average value for the same day of the week over the last 4 weeks, same time of day.

    :param ts: Timestamp of the forecast
    :return: Average PV yield (kWh) for the same day of the week in the last 4 weeks or None
    """
    vals: list[float] = []
    for w in range(1, 5):
        y = ts - timedelta(weeks=w)
        v = get_hourly_yield_by_timestamp(y.strftime("%Y-%m-%d"), y.strftime("%H:%M:%S"))
        if v is not None:
            vals.append(v)

    return float(sum(vals) / len(vals)) if vals else None


# ----------------------------
# Daily baselines
# ----------------------------
def baseline_daily_y_minus_1d(ts: datetime) -> float | None:
    """
    Takes the previous day's daily value as the baseline.

    :param ts: Timestamp of the forecast
    :return: PV yield (kWh) of the previous day or None
    """
    y = ts - timedelta(days=1)

    return get_daily_yield_by_timestamp(y.strftime("%Y-%m-%d"))


def baseline_daily_mean_last7d(ts: datetime) -> float | None:
    """
    Average value of the last 7 days (daily values) as baseline.

    :param ts: Timestamp of the forecast
    :return: Average daily yield (kWh) of the last 7 days or None
    """
    vals: list[float] = []
    for d in range(1, 8):
        y = ts - timedelta(days=d)
        v = get_daily_yield_by_timestamp(y.strftime("%Y-%m-%d"))
        if v is not None:
            vals.append(v)

    return float(sum(vals) / len(vals)) if vals else None


def baseline_daily_same_dow_last4w_mean(ts: datetime) -> float | None:
    """
    Average value of the last 7 days (daily values) as baseline.

    :param ts: Timestamp of the forecast
    :return: Average daily yield (kWh) of the last 7 days or None
    """
    vals: list[float] = []
    for w in range(1, 5):
        y = ts - timedelta(weeks=w)
        v = get_daily_yield_by_timestamp(y.strftime("%Y-%m-%d"))
        if v is not None:
            vals.append(v)

    return float(sum(vals) / len(vals)) if vals else None
