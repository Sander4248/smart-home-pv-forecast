from __future__ import annotations
import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib import irradiance, pvsystem

from src.config.schema import RunCfg, ProjectParams


def _clearsky_core(cfg: RunCfg, times: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute the clear-sky PV yield (hourly) for the project-defined arrays.

    :param cfg: Central configuration instance for the training run with all parameters
    :param times: Timestamps for which the clear-sky calculation should be performed.
    :return: DataFrame with columns:
             - datetime: pd.DatetimeIndex (naive, local, without TZ)
             - P_cs_kW:  float, computed clear-sky power in kW (sum of all arrays)
             - P_cs_kWh: float, simplified conversion kW → kWh (valid for 1h time grid)
    """
    # Create a site object for the given LAT/LON/TZ
    site = Location(cfg.params.LATITUDE, cfg.params.LONGITUDE, cfg.params.TIMEZONE)

    # Calculate solar position (zenith, azimuth etc.) for the time series
    solpos = site.get_solarposition(times)

    # Clear-sky global/diffuse/direct irradiance (GHI, DHI, DNI) using the Ineichen model
    cs = site.get_clearsky(times, model="ineichen")

    # Prepare array for aggregated PV power
    p_cs_kw = np.zeros(len(times), dtype=float)

    # Compute and sum up all defined arrays (e.g., east/south/west roofs)
    for arr in cfg.params.STRINGS:
        # Irradiance on module surface (Plane of Array, POA)
        poa = irradiance.get_total_irradiance(
            surface_tilt=cfg.params.TILT_DEG,
            surface_azimuth=arr.azimuth,
            dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"],
            solar_zenith=solpos["apparent_zenith"],
            solar_azimuth=solpos["azimuth"],
        )

        # PVWatts model: DC power at STC-like temperature (simplified assumption: 25 °C)
        p_dc = pvsystem.pvwatts_dc(
            effective_irradiance=poa["poa_global"],  # irradiance in W/m²
            temp_cell=pd.Series(cfg.params.pvwatts_temp, index=times),
            pdc0=arr.kwp * 1000.0,  # module capacity in W
            gamma_pdc=cfg.params.pvwatts_gamma,  # temperature coefficient
        )

        # Convert W → kW and add to total
        p_cs_kw += (p_dc / 1000.0).to_numpy()

    # DataFrame with local-naive timestamps and clear-sky power
    df = pd.DataFrame({"datetime": times.tz_localize(None), "P_cs_kWh": p_cs_kw})

    return df


def clearsky_hourly(cfg: RunCfg, times: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hourly clear-sky yields for arbitrary input times.

    :param cfg: Central configuration instance for the training run with all parameters
    :param times: Timestamps (local-naive or tz-aware).
                  Local-naive times are automatically interpreted as Europe/Berlin.
    :return: DataFrame with columns:
             - datetime: datetime64[ns] (local-naive, without TZ)
             - P_cs_kWh: float, clear-sky yield per hour [kWh]
    """
    # Robust conversion: works with Series, Index or list
    idx = pd.DatetimeIndex(times)

    # Local-naive input → localize to Europe/Berlin
    if idx.tz is None:
        idx = idx.tz_localize(cfg.params.TIMEZONE)
    else:
        # If already tz-aware → convert to Europe/Berlin
        idx = idx.tz_convert(cfg.params.TIMEZONE)

    # Core calculation (returns local-naive)
    return _clearsky_core(cfg, idx)[["datetime", "P_cs_kWh"]]


def clearsky_daily(cfg: RunCfg, dates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily clear-sky yields (sum per day).

    :param cfg: Central configuration instance for the training run with all parameters
    :param dates: Date values (local-naive). Can be Series, array or list.
    :return: DataFrame with columns:
             - date:     datetime64[ns] (day, local-naive)
             - P_cs_kWh: float, daily clear-sky yield [kWh
    """
    # Extract unique day values
    uniq = pd.to_datetime(pd.Series(dates)).dt.date.unique()

    # Build an hourly time grid (TZ-aware, Europe/Berlin) for each day
    all_hours = [
        pd.date_range(f"{d} 00:00:00", f"{d} 23:00:00", freq="1h", tz=cfg.params.TIMEZONE)
        for d in uniq
    ]

    # Special case: empty input
    if len(all_hours) == 0:
        return pd.DataFrame(columns=["date", "P_cs_kWh"])

    # Concatenate all hourly grids (keeps TZ-aware)
    all_times = all_hours[0].append(all_hours[1:])

    # Core calculation (expects TZ-aware and returns local-naive)
    hourly = _clearsky_core(cfg, all_times)

    # Derive date column robustly (normalize removes time of day)
    dt_col = hourly["datetime"]
    date_naiv = pd.to_datetime(dt_col).dt.normalize()
    hourly["date"] = date_naiv

    # Aggregate per day
    daily = hourly.groupby("date", as_index=False)["P_cs_kWh"].sum()
    return daily[["date", "P_cs_kWh"]]
