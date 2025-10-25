import logging
import os
import sqlite3

from src.config.logging_config import setup_logging
from src.config.paths import RESULTS_DB_PATH


def main():
    """
    Creates the database ‘training_results’ and all necessary tables.

    If the database already exists, it will be deleted and recreated.
    """

    setup_logging()

    # Questions about whether the database should be reset
    if os.path.exists(RESULTS_DB_PATH):
        confirm = input(f"Database '{RESULTS_DB_PATH}' already exists. Really want to delete? (yes/no): ")

        # Only delete if “yes” has been explicitly entered
        if confirm.strip().lower() == "yes":
            os.remove(RESULTS_DB_PATH)
            logging.warning(f"Deleted database '{RESULTS_DB_PATH}'")
        else:
            logging.info(f"Aborted. Database '{RESULTS_DB_PATH}' was not deleted.")
            exit(1)

    # Connect to the database or create a new one
    with sqlite3.connect(RESULTS_DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_evaluate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER NOT NULL,
                start TEXT,
                end TEXT,
                months TEXT,
                model_label TEXT,
                night_clamp BOOLEAN NOT NULL,
                night_clamp_threshold REAL NOT NULL,
                forecast_horizon INTEGER NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                nrmse REAL NOT NULL,
                bias_me REAL NOT NULL,
                r2 REAL NOT NULL,
                smape REAL NOT NULL,
                mape_y REAL NOT NULL,
                forecast_sum REAL NOT NULL,
                actual_sum REAL NOT NULL,
                energy_balance_error REAL NOT NULL,
                lt_25 REAL NOT NULL,
                lt_50 REAL NOT NULL,
                lt_75 REAL NOT NULL,
                lt_100 REAL NOT NULL,
                gt_100 REAL NOT NULL,
                inf REAL NOT NULL,
                FOREIGN KEY (result_id) REFERENCES daily_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER,
                model BLOB,
                FOREIGN KEY (result_id) REFERENCES daily_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_rank_forward_selection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER,
                rank INTEGER,
                feature_name TEXT,
                FOREIGN KEY (result_id) REFERENCES daily_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                mae REAL,
                period TEXT NOT NULL,
                months TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES session_settings(session_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_evaluate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER NOT NULL,
                start TEXT,
                end TEXT,
                months TEXT,
                model_label TEXT,
                night_clamp BOOLEAN NOT NULL,
                night_clamp_threshold REAL NOT NULL,
                forecast_horizon INTEGER NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                nrmse REAL NOT NULL,
                bias_me REAL NOT NULL,
                r2 REAL NOT NULL,
                smape REAL NOT NULL,
                mape_y REAL NOT NULL,
                forecast_sum REAL NOT NULL,
                actual_sum REAL NOT NULL,
                energy_balance_error REAL NOT NULL,
                n_hours REAL NOT NULL,
                n_daylight_hours REAL NOT NULL,
                n_inf REAL NOT NULL,
                lt_25 REAL NOT NULL,
                lt_50 REAL NOT NULL,
                lt_75 REAL NOT NULL,
                lt_100 REAL NOT NULL,
                gt_100 REAL NOT NULL,
                inf REAL NOT NULL,
                count_days REAL NOT NULL,
                median_h REAL NOT NULL,
                p90_h REAL NOT NULL,
                min_h REAL NOT NULL,
                max_h REAL NOT NULL,
                FOREIGN KEY (result_id) REFERENCES hourly_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER,
                model BLOB,
                FOREIGN KEY (result_id) REFERENCES hourly_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_rank_forward_selection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER,
                rank INTEGER,
                feature_name TEXT,
                FOREIGN KEY (result_id) REFERENCES hourly_results(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                mae REAL,
                period TEXT NOT NULL,
                months TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES session_settings(session_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                feature_name TEXT,
                FOREIGN KEY (session_id) REFERENCES session_settings(session_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_settings (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                test_size FLOAT NOT NULL,
                test_size_shuffle BOOLEAN NOT NULL,
                n_estimators FLOAT NOT NULL,
                max_depth FLOAT NOT NULL,
                learning_rate FLOAT NOT NULL,
                subsample FLOAT NOT NULL,
                colsample_bytree FLOAT NOT NULL, 
                reg_lambda FLOAT NOT NULL, 
                random_state INTEGER NOT NULL,
                data_resolution TEXT NOT NULL,
                forward_selection BOOLEAN NOT NULL,
                use_clearsky_feature BOOLEAN NOT NULL,
                normalize_by_clearsky BOOLEAN NOT NULL,
                limit_training_data BOOLEAN NOT NULL,
                eval_with_weather_history_data BOOLEAN NOT NULL,
                description TEXT
            )
        """)

        conn.commit()

    logging.info(f"Database '{RESULTS_DB_PATH}' has been reset.")

if __name__ == "__main__":
    main()
