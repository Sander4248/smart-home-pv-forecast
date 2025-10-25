import logging
from dataclasses import dataclass, field
from typing import Literal, List

"""
------------------------------------------------------------------------------------------------------------------------
ModelParams (Model-specific hyperparameters)
------------------------------------------------------------------------------------------------------------------------
- type:              Defines the type of model to be trained (XGBRegressor, RandomForestRegressor).
- test_size:         Size of the test split. A value of 0 means no test split, only training.
- test_size_shuffle: Determines whether the dataset should be randomly shuffled before splitting.
- n_estimators:      Number of boosting iterations (trees) used in training.
- max_depth:         Maximum depth of individual decision trees. 
                     Higher values increase model complexity and risk of overfitting.
- random_state:      Seed for reproducibility of results.

XGBRegressor exclusive params (ignored in RandomForestRegressor)
----------------------------------------------------------------
- learning_rate:     Step size for updating new trees. Smaller values make training more stable but slower.
- subsample:         Proportion of training samples randomly selected for each tree. 
                     Values below 1 act as regularization.
- colsample_bytree:  Proportion of features randomly selected for each tree.
- reg_lambda:        L2 regularization weight to prevent overfitting.


------------------------------------------------------------------------------------------------------------------------
TrainingCfg (Training-specific settings)
------------------------------------------------------------------------------------------------------------------------
- data_resolution:       Resolution of the dataset, either hourly or daily. 
                         For daily data, some features (such as azimuth or hour_sin) are excluded automatically.
- forward_selection:     If enabled, applies forward feature selection to identify the most predictive features.
                         If disabled, trains using all specified features.
- limit_training_data:   Limits the training data to the period for which no weather forecast data is yet available. 
                         Must be activated for auto evaluation. Otherwise, days that were also trained are checked 
                         during evaluation, which is not possible with a forecast model.
- use_clearsky_feature:  Adds the calculated clear sky yield 'P_cs_kWh' to the training features. 
                         If auto evaluation is enabled, the clear sky index is also included here.
- normalize_by_clearsky: Normalizes PV yield based on the clear sky index. 
                         The threshold value 'EPS_PCS' can be set in params.py. 
                         Requires 'use_clearsky_feature' = true (automatically checked and set if necessary).
- months:                Defines the months used for training. 
                         A flat list runs a single training session with those months. 
                         A nested list runs multiple sessions for different month groups (e.g., seasons). 
                         To train on a full year, all months must be included in a nested list.


------------------------------------------------------------------------------------------------------------------------
EvalCfg (Evaluation parameters)
------------------------------------------------------------------------------------------------------------------------
- auto_eval:                      Determines whether evaluation is automatically performed after training.
- night_clamp:                    Indicates whether values during nighttime should be clamped or adjusted.
- night_clamp_threshold:          Threshold value used for nighttime clamping. 
                                  Predictions below this threshold during night are adjusted.
- store:                          Determines whether evaluation results are stored.
- eval_with_weather_history_data: For testing purposes only!
                                  If enabled, historical instead forecast weather data is used for evaluation.
- forecast_horizon:               Defines the forecasting horizon in days between the target and prediction time.
- aggregate_hourly_to_daily:      Experimental! Aggregate hourly forecasts into daily totals.


------------------------------------------------------------------------------------------------------------------------
MetaCfg (Meta information)
------------------------------------------------------------------------------------------------------------------------
description: Optional description for the current training session, useful for documentation or logging.


------------------------------------------------------------------------------------------------------------------------
FeaturesCfg (Input Features)
------------------------------------------------------------------------------------------------------------------------
Note: The three feature groups weather, solar_pos, and timestamp must always be present (possibly as empty lists).
- weather: List of weather features to be used.
- solar_pos: List of solar position features (e.g., apparent_elevation).
- timestamp: List of time-based features.


------------------------------------------------------------------------------------------------------------------------
StringCfg (Per-String Orientation)
------------------------------------------------------------------------------------------------------------------------
- azimuth: Azimuth of the module surface in degrees (0° = North, 90° = East, 180° = South, 270° = West).
- kwp: Rated power of the string in kWp. Used to scale expected PV yields


------------------------------------------------------------------------------------------------------------------------
ProjectParams (Location & System Parameters)
------------------------------------------------------------------------------------------------------------------------
- LATITUDE: Geographic latitude of the site in degrees (required for solar position and clearsky calculations).
- LONGITUDE: Geographic longitude of the site in degrees (required for solar position and clearsky calculations).
- START_DATE: str (YYYY-MM-DD): Start date of the historical data basis (e.g., for training/feature calculation).
- START_FORECAST_DATE: str (YYYY-MM-DD): First date from which forecast data (features) are used.
- TIMEZONE: IANA timezone of the site (e.g., Europe/Berlin); affects timestamp features and daylight logic.
- TILT_DEG: Roof/module tilt in degrees. Used for yield or irradiance models.
- STRINGS: List of string/module alignments and their respective kWp (see StringCfg). 
           Allows multiple orientations within one system.
- end_date: str (YYYY-MM-DD): End date for training/evaluation. Depends on the status of the databases.
- eps_pcs: Threshold for clear-sky normalization (kWh). 
           Yield normalization only applies once this clear-sky potential has been reached.
- pvwatts_gamma: Temperature coefficient of power (per °C) for PVWatts-like estimation 
                 (conservative approach, typically negative).
- pvwatts_temp: Reference cell temperature in °C for PVWatts estimation.


------------------------------------------------------------------------------------------------------------------------
RunCfg (Overall configuration of a training run)
------------------------------------------------------------------------------------------------------------------------
- model:      Contains the model hyperparameters.
- training:   Contains the training configuration.
- evaluation: Contains the evaluation configuration.
- features:   Contains the feature configuration.
- params:     Contains project-specific location and system parameters.
- meta:       Contains metadata such as a description of the run.


------------------------------------------------------------------------------------------------------------------------
Rules for months:
------------------------------------------------------------------------------------------------------------------------
 - Flat list [ ... ] → one model per month
 - Nested list [[ ... ], [ ... ], ...] → one model per group
 - Values must be 1–12 (Jan–Dec)

Examples & behavior
-------------------
months = [1,2,3,4,5,6,7,8,9,10,11,12]
         Trains 12 models, one per month (January … December).
         Use when you want highly month-specific models.

months = [1,2]
         Trains 2 models, one for January and one for February.
         Use when you only need specific months, each as its own model.

months = [[1,2,3,4,5,6,7,8,9,10,11,12]]
         Trains 1 model using all months (annual model).
         Use when you want a single model that generalizes across the year.

months = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
         Trains 4 models, one per season group (Winter, Spring, Summer, Autumn as defined).
         Use when seasonality differs significantly.

months = [[1,2,3],[4,5,6]]
         Trains 2 models, custom seasonal groupings (Q1-like vs. Q2-like).
         Use for bespoke seasonal splits.

months = [7]
         Trains 1 model for July only.
         Use for a single-month focus (e.g., a pilot or targeted deployment).

Practical tips & edge cases
---------------------------
- Ordering doesn't matter within a group: [12,1,2] ≡ [1,2,12].
- Avoid duplicates; if present, treat them as one (e.g., [1,1,2] → months 1 and 2).
- Don't mix flat and nested formats in one config (e.g., [1, [2,3]]): pick one style.
- Overlapping groups are allowed but will train multiple models using overlapping data (be intentional).
- Coverage strategy:
  - Fine-grained: flat list covering many months → many specialized models.
  - Seasonal: nested 3–4 groups → balanced complexity vs. robustness.
  - Global: single group with all months → simplest, most data, least seasonal specialization.
"""


@dataclass
class StringCfg:
    azimuth: float
    kwp: float


@dataclass
class ProjectParams:
    LATITUDE: float
    LONGITUDE: float
    START_DATE: str
    START_FORECAST_DATE: str
    TIMEZONE: str
    TILT_DEG: float
    STRINGS: List[StringCfg]
    end_date: str
    eps_pcs: float
    pvwatts_gamma: float
    pvwatts_temp: float


@dataclass
class ModelParams:
    type: str  # XGBRegressor or RandomForestRegressor
    # XGBRegressor and RandomForestRegressor params
    test_size: float
    test_size_shuffle: bool
    n_estimators: int
    max_depth: int
    random_state: int
    # XGBRegressor only (Ignored in other models)
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float


@dataclass
class TrainingCfg:
    data_resolution: str
    forward_selection: bool
    limit_training_data: bool
    use_clearsky_feature: bool
    normalize_by_clearsky: bool
    months: List[int]


@dataclass
class EvalCfg:
    auto_eval: bool
    night_clamp: bool
    night_clamp_threshold: float
    store: bool
    eval_with_weather_history_data: bool
    forecast_horizon: int
    aggregate_hourly_to_daily: bool


@dataclass
class FeaturesCfg:
    weather: List[str]
    solar_pos: List[str]
    timestamp: List[str]


@dataclass
class MetaCfg:
    description: str


@dataclass
class RunCfg:
    model: ModelParams
    training: TrainingCfg
    evaluation: EvalCfg
    features: FeaturesCfg
    params: ProjectParams
    meta: MetaCfg

    def __post_init__(self):

        if self.training.months is None:
            logging.critical("No training period specified. Training not possible.")
            exit(1)

        if self.evaluation.aggregate_hourly_to_daily:
            logging.warning("Attention! Experimental function 'aggregate_hourly_to_daily' is activated!")
            if self.training.data_resolution == "daily" or self.evaluation.store:
                logging.critical("Experimental function xyt is activated. For this, 'data_resolution' must be "
                                 "set to 'hourly' and 'store' must be set to 'false'!")
                exit(1)

        if self.model.test_size == 0 and self.training.forward_selection:
            logging.error("Forward selection is only possible if training data is split (model.test_size > 0). "
                          "Forward selection is disabled.")
            self.training.limit_training_data = False

        if self.evaluation.auto_eval and self.evaluation.night_clamp and self.training.data_resolution == "daily":
            logging.error("Night clamp is only useful when evaluating hourly models. Has been deactivated.")
            self.evaluation.night_clamp = False

        if not (0 <= self.evaluation.forecast_horizon <= 16):
            logging.error("Evaluation forecast horizon must be between 0 and 16 (inclusive). Has been set to 1.")
            self.evaluation.forecast_horizon = 1

        if self.training.normalize_by_clearsky and not self.training.use_clearsky_feature:
            logging.error("The normalize by clearsky option can only be used effectively if the clear sky features "
                          "are used for training. The clear sky feature has been activated.")
            self.training.use_clearsky_feature = True

        if self.model.test_size == 0:
            logging.warning("Testing size is 0. Training data is not split.")

        if self.evaluation.auto_eval and not self.training.limit_training_data:
            logging.warning("Evaluation is enabled even though the training data is not limited. The same PV "
                            "yield data can be used for training and evaluation, which leads to distorted results.")

        if self.evaluation.eval_with_weather_history_data:
            logging.warning("Attention! For testing purposes only: 'eval_with_weather_history_data' is enabled. "
                            "Historical weather data (instead of forecast data) is used for evaluation.")

        logging.info("  Training Parameters")

        if self.meta.description == "":
            logging.info("  Training description: No training description provided.")
        else:
            logging.info(f"  Training description: {self.meta.description}")

        logging.info(f"  ML model: {self.model}")
        logging.info(f"  Data resolution: {self.training.data_resolution}")

        cs_status = "On" if self.training.use_clearsky_feature else "Off"
        logging.info(f"  Clear sky: {cs_status}")

        n_status = "On" if self.training.normalize_by_clearsky else "Off"
        logging.info(f"  Yield normalization: {n_status}")

        fs_status = "On" if self.training.forward_selection else "Off"
        logging.info(f"  Forward selection: {fs_status}")

        ae_status = "On" if self.evaluation.auto_eval else "Off"
        logging.info(f"  Auto evaluation: {ae_status}")

        if self.evaluation.auto_eval:
            logging.info(f"  Forecast horizon: {self.evaluation.forecast_horizon}")

            nc_staus = "On" if self.evaluation.night_clamp else "Off"
            text = f"  Night clamp: {nc_staus}"
            if self.evaluation.night_clamp:
                text += f", with {self.evaluation.night_clamp_threshold} degree threshold"
            logging.info(text)

            store_status = "Yes" if self.evaluation.store else "No"
            logging.info(f"  Save evaluation results in database: {store_status}")
