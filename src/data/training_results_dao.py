from __future__ import annotations

import pickle
from datetime import datetime
from typing import List, Tuple, Any

from xgboost import XGBModel, XGBRegressor

from src.config.paths import RESULTS_DB_PATH
from src.config.schema import RunCfg
from src.utils import get_cursor

SESSION_ID: int  # Stores the ID of the currently active session after insertion
RESULT_ID: int  # Stores the ID of the currently active session after insertion

# ----------------------------
# Getter
# ----------------------------

def get_session_id() -> int:
    """
    Determines the last used session ID.

    :return: Highest existing session ID
    """
    with get_cursor(RESULTS_DB_PATH) as cursor:
        query = """
            SELECT MAX(session_id)
              FROM session_settings
        """
        cursor.execute(query)
        session_id = cursor.fetchone()[0]
    return session_id


def get_results_data(data_resolution: str,
                     session_id: int) -> List[Tuple[int, str]]:
    """
    Get all result IDs from the last session (each result ID represents a trained model), as well as the monthly
    figures used to train each model.

    :param data_resolution: Resolution of the dataset. Must be either "hourly" or "daily"
    :param session_id: Session ID to load
    :return: List of (result_id, months_string), where 'months_string' can be, for example, “1;2;3”
    """
    with get_cursor(RESULTS_DB_PATH) as cursor:
        query = f"""
            SELECT m.result_id, r.months
              FROM {data_resolution}_models m
              LEFT JOIN {data_resolution}_results r
                ON m.result_id = r.id
              LEFT JOIN session_settings s
                ON r.session_id = s.session_id
             WHERE r.session_id = {session_id}
        """

        cursor.execute(query)
        results = cursor.fetchall()

    return results


def load_model_from_db(data_resolution: str,
                       result_id: int
                       ) -> Any:
    """
    Loads a saved model from the SQLite database.

    :param data_resolution: Prefix for table selection (e.g., 'hourly' or 'daily')
    :param result_id: Unique ID of the stored model
    :return: Deserialized trained model (XGBRegressor)
    :raise: ValueError if no model with the specified 'result_id' is found in the table.
    """
    query = f"""
        SELECT model
          FROM {data_resolution}_models
         WHERE result_id = {result_id}
    """
    with get_cursor(RESULTS_DB_PATH) as cursor:
        cursor.execute(query)
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"No model found for result_id={result_id}.")
        return pickle.loads(row[0])

# ----------------------------
# Setter
# ----------------------------

def save_session(cfg: RunCfg, features: list[str]):
    """
    Stores metadata of a training session, including:
     - hyperparameters (train_params),
     - selected features,
     - whether forward feature selection was used,
     - and a textual description

    Stores the session ID created by DB globally.

    Stores one row in SESSION_SETTINGS and one row per feature in SESSION_FEATURES.
    :param cfg: Central configuration instance for the training run with all parameters
    :param features: List of selected feature names.
    """
    global SESSION_ID

    with get_cursor(RESULTS_DB_PATH) as cursor:
        query, values = _get_session_settings_statement(cfg)
        cursor.execute(query, values)
        SESSION_ID = cursor.lastrowid

        for feature in features:
            query, values = _get_session_features_statement(feature)
            cursor.execute(query, values)

        pass


def save_training_result(mae: float, db_insert_para):
    """
    Saves a model’s result for a session.

    Stores one row in the results table and updates the global RESULT_ID.
    :param mae: Mean Absolute Error of the trained model.
    :param db_insert_para: Dictionary with keys:
            - 'table_name_results': table name to insert the result,
            - 'period': label for the time period (e.g., "Summer").
    """
    global RESULT_ID
    query = f"""
        INSERT INTO {db_insert_para['table_name_results']} (session_id, mae, period, months)
        VALUES (?, ?, ?, ?)
    """
    values = (SESSION_ID, mae, db_insert_para['period'], db_insert_para['months'])
    with get_cursor(RESULTS_DB_PATH) as cursor:
        cursor.execute(query, values)
        RESULT_ID = cursor.lastrowid


def save_model(model: Any, db_insert_para: dict):
    """
    Saves a trained model to the database.

    If forward selection is on, only the best model would be saved.
    :param model: Trained model
    :param db_insert_para: Must contain 'table_name_ranking'.
    """
    query = f"""
        INSERT INTO {db_insert_para['table_name_model']} (result_id, model)
        VALUES (?, ?)
    """
    values = (RESULT_ID, pickle.dumps(model))
    with get_cursor(RESULTS_DB_PATH) as cursor:
        cursor.execute(query, values)


def save_ranking(feature: str, rank: int, db_insert_para: dict):
    """
    Saves a single feature's ranking for a result.

    Stores one row in the ranking table.
    :param feature: Name of the feature.
    :param rank: Rank of the feature.
    :param db_insert_para: Must contain 'table_name_ranking'.
    """
    query = f"""
        INSERT INTO {db_insert_para['table_name_ranking']} (result_id, rank, feature_name)
        VALUES (?, ?, ?)
    """
    values = (RESULT_ID, rank, feature)
    with get_cursor(RESULTS_DB_PATH) as cursor:
        cursor.execute(query, values)


def _get_session_features_statement(feature: str):
    """
    Helper function that prepares the SQL insert statement for a single feature.
    :param feature: Name of the feature.
    :return: Insert query and values for a single feature.
    """
    query = """
        INSERT INTO SESSION_FEATURES (session_id, feature_name)
        VALUES (?, ?)
    """
    values = (SESSION_ID, feature)
    return query, values


def _get_session_settings_statement(cfg: RunCfg):
    """
    Helper function to build the insert query for session settings.

    :param cfg: Central configuration instance for the training run with all parameters
    :return: Insert query and values for session settings.
    """
    query = """
        INSERT INTO SESSION_SETTINGS (timestamp, test_size, test_size_shuffle, n_estimators, max_depth, learning_rate, 
        subsample, colsample_bytree, reg_lambda, random_state, data_resolution, forward_selection, description,
        use_clearsky_feature, normalize_by_clearsky, limit_training_data, eval_with_weather_history_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cfg.model.test_size,
        cfg.model.test_size_shuffle,
        cfg.model.n_estimators,
        cfg.model.max_depth,
        cfg.model.learning_rate,
        cfg.model.subsample,
        cfg.model.colsample_bytree,
        cfg.model.reg_lambda,
        cfg.model.random_state,
        cfg.training.data_resolution,
        cfg.training.forward_selection,
        cfg.meta.description,
        cfg.training.use_clearsky_feature,
        cfg.training.normalize_by_clearsky,
        cfg.training.limit_training_data,
        cfg.evaluation.eval_with_weather_history_data
    )
    return query, values


def save_evaluation_results(data_resolution: str,
                            db_data: dict):
    """
    Store the processed evaluation results in the database.

    :param data_resolution: Resolution of the dataset to select the correct table
    :param db_data: Dictionary with data that should be persisted
    """

    columns = ",".join(db_data.keys())
    placeholders = ",".join(["?"] * len(db_data))
    query = f"INSERT INTO {data_resolution}_evaluate ({columns}) VALUES ({placeholders})"
    with get_cursor(RESULTS_DB_PATH) as cursor:
        cursor.execute(query, tuple(db_data.values()))
