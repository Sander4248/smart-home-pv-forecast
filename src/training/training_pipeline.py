import logging
from dataclasses import asdict

from src.config.schema import RunCfg
from src.data.training_results_dao import save_session, save_training_result, save_model
from src.training.load_data import build_dataframe
from src.training.model_trainer import _forward_select_features, train_model
from src.training.utils import _logging_months, get_db_params


def run_training_pipeline(cfg: RunCfg):
    """
    Orchestrate the end-to-end training process.

    This function prepares features, records session metadata, optionally augments the dataset with clear-sky
    features, supports normalization by clear-sky energy, and trains a model either directly or via forward
    feature selection. It supports both hourly and daily data resolutions and iterates over multiple months
    configured in 'cfg.training.months'.

    Notes:
      - If 'cfg.training.use_clearsky_feature' is True, the clear-sky energy 'P_cs_kWh' is appended as an input feature.
      - If 'cfg.training.normalize_by_clearsky' is True, the target 'pv_yield' is replaced by the normalized target
        'pv_norm' produced during dataframe building.
      - Some features are excluded because they are not meaningful for training in the given context
        (e.g., categorical weather codes or hour-of-day for daily models).
      
    :param cfg: Central configuration instance for the training run with all parameters
    """

    logging.info(f"Start training model/s.")

    # Flatten grouped features into a single list (weather + solar_pos + timestamp)
    # training_features_list = [feature for sublist in features.values() for feature in sublist]
    training_features_list = (cfg.features.weather + cfg.features.solar_pos + cfg.features.timestamp)

    # Exclude features that are not useful for training the target (domain decision)
    excluded_features = ['weather_code']
    training_features_list = [f for f in training_features_list if f not in excluded_features]

    # Optionally include clear-sky potential as an input feature
    if cfg.training.use_clearsky_feature:
        training_features_list.append('P_cs_kWh')

    # For daily models, drop features that are only meaningful hourly (e.g., hour-of-day, per-timestamp azimuth)
    if cfg.training.data_resolution == 'daily':
        excluded_daily_features = ['azimuth', 'hour_sin', 'hour_cos']
        training_features_list = [f for f in training_features_list if f not in excluded_daily_features]

    # Persist session metadata (e.g., to link results to the current configuration + feature set)
    save_session(cfg, training_features_list)

    # Train separately for each month (helps detect seasonality and avoids leakage across months)
    for month in cfg.training.months:
        _logging_months(month)

        # Build the dataframe for this month and the requested feature groups
        df = build_dataframe(cfg, month)
        logging.debug(f"Number of rows in the training dataframe: {df.shape[0]}")

        # If normalization is enabled, switch target to pv_norm (produced by _normalize_dataframe)
        # The rest of the pipeline continues to use the column name 'pv_yield' as the target.
        if cfg.training.normalize_by_clearsky:
            df = df.copy()
            df["pv_yield"] = df["pv_norm"]  # replace target with normalized target

        # Prepare identifiers/keys for persistence (per-resolution, per-month)
        db_insert_params = get_db_params(cfg.training.data_resolution, month)

        if cfg.training.forward_selection:
            # Perform forward selection using a copy of the feature list to preserve original order
            _forward_select_features(training_features_list.copy(), df, asdict(cfg.model), db_insert_params)
        else:
            # Train a single model with the provided features
            mae, model = train_model(training_features_list, df, asdict(cfg.model))

            # Persist metric and model artifact
            save_training_result(mae, db_insert_params)
            save_model(model, db_insert_params)

            # Log metric with unit depending on normalization mode
            if mae:
                unit = "kWh" if not cfg.training.normalize_by_clearsky else "normalized"
                logging.info(f"MAE: {mae:.3f} {unit}")

    logging.info("Training completed.")
