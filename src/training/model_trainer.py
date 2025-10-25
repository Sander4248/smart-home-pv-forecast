import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data.training_results_dao import save_training_result, save_model, save_ranking


def train_model(
        base_features: list,
        df: pd.DataFrame,
        params: dict,
) -> (None | float, XGBRegressor | RandomForestRegressor):
    """
    Trains an XGBRegressor or RandomForestRegressor model to predict PV yield.

    If test size > 0 the mean absolute error (MAE) would be returned.

    :param base_features: List of desired input features. Only features present in the DataFrame's columns will be used.
    :param df: Training dataset that must include the input features and the target variable 'pv_yield'.
    :param params: Parameters for model training. Expected keys: test_size, n_estimators, max_depth, learning_rate
    :return: Rounded mean absolute error (MAE) of the model on the test dataset (if test size > 0) and model.
    """
    # Update feature list
    feature_cols = [f for f in base_features if f in df.columns]
    x = df[feature_cols]

    # Set target variable
    y = df["pv_yield"]

    # Train/test split
    if params["test_size"] == 0:
        x_train = x
        y_train = y
    else:
        x_train, x_test, y_train, y_test = (
            train_test_split(x, y, shuffle=params["test_size_shuffle"], test_size=params["test_size"]))

    # Training
    if params["type"] == "XGBRegressor":
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            random_state=params["random_state"]
        )
    elif params["type"] == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=-1  # All available cores of the system are used.
        )
    else:
        logging.critical(f"'{params["type"]}' is an not supported regressor for model training. Aborted.'")
        exit(1)
    model.fit(x_train, y_train)

    # Feature importance's
    if params["type"] == "XGBRegressor":
        importance = model.get_booster().get_score(importance_type='gain')
        logging.debug("Feature importances (gain):")
        s = pd.Series({k: float(v) for k, v in importance.items()}).sort_values(ascending=False)
        for row in s.to_string(float_format=lambda x: f"{x:.6f}").split("\n"):
            logging.debug(row)

    # Evaluation
    if params["test_size"] == 0:
        return None, model
    else:
        y_pred = model.predict(x_test)
        model_metrics = _eval_metrics(y_test, y_pred)

        print(f"{params["type"]},{params["test_size"]},,,,{model_metrics["MAE"]:.2f},{model_metrics["R2"]:.2f},"
              f",{model_metrics["RMSE"]:.2f},{model_metrics["nRMSE_range"]:.2f},{model_metrics["Bias_ME"]:.2f},{model_metrics["sMAPE_%"]:.2f},")

        return model_metrics["MAE"], model

# ---------- Metriken ----------
def bias_me(y_true: pd.Series, y_pred: pd.Series) -> float:
    # Mean Error (positiv = Überschätzung)
    return float(np.mean(y_pred - y_true))

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def nrmse_range(y_true: pd.Series, y_pred: pd.Series) -> float:
    rng = float(y_true.max() - y_true.min()) or 1.0
    return rmse(y_true, y_pred) / rng

def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)).replace(0, np.finfo(float).eps)
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / denom))

def _eval_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:

    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "nRMSE_range": nrmse_range(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias_ME": bias_me(y_true, y_pred),
        "sMAPE_%": smape(y_true, y_pred),
    }

    # for key, value in metrics.items():
    #     logging.debug(f"{key}: {value:.3f}")

    return metrics

def _forward_select_features(training_features: list, df: pd.DataFrame, train_params: dict, db_insert_params: dict):
    """
    Performs forward feature selection to determine the best input features for PV yield prediction.

    The function iteratively tests all remaining features in combination with already selected ones, and
    selects the feature that yields the lowest mean absolute error (MAE). The process stops when no further
    improvement in MAE is achieved.

    The List of selected features in the order they were added will be logged.
    :param training_features: List of available features from which the best subset will be selected.
    :param df: Training dataset that must include the input features and the target variable 'pv_yield'.
    :param train_params: Parameters for model training. Expected keys: test_size, n_estimators, max_depth, learning_rate
    :param db_insert_params: Dictionary of parameters for database insertion.
    """

    # Initialize tracking for selected features and MAE
    forward_selection_list = []
    rank = 0  # Ranking of features
    latest_mae = 1000  # Arbitrarily high to ensure first MAE is lower

    # Loop until no features left
    while training_features:
        rank += 1
        maes = []  # Store (MAE, feature set) for each candidate feature

        # Test each remaining feature in combination with selected ones
        for base_feature in training_features:
            train_features = forward_selection_list + [base_feature]
            current_mae, current_model = train_model(train_features, df, train_params)
            maes.append((current_mae, train_features))

        # Pick feature set with lowest MAE
        maes_sorted = sorted(maes, key=lambda x: x[0])
        best_mae_tuple = maes_sorted[0]
        current_mae, best_mae_features = best_mae_tuple
        latest_added_feature = best_mae_features[-1]

        # Add best new feature and remove from pool
        forward_selection_list.append(latest_added_feature)
        training_features.remove(latest_added_feature)

        # Stop if MAE didn't improve
        if current_mae >= latest_mae:
            # Train model with best features again and store in DB
            mae, model = train_model(forward_selection_list[:-1], df, train_params)
            save_model(model, db_insert_params)
            break
        else:
            logging.info(f"{best_mae_tuple}")
            save_training_result(current_mae, db_insert_params)
            save_ranking(latest_added_feature, rank, db_insert_params)
            latest_mae = current_mae
