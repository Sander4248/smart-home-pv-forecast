import logging

from src.config.logging_config import setup_logging
from src.config.paths import RESULTS_DB_PATH
from src.data.training_results_dao import load_model_from_db
from src.utils import get_cursor


def main():
    with get_cursor(RESULTS_DB_PATH) as cursor:
        query = """
            SELECT result_id
              FROM hourly_models
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        result_ids = [row[0] for row in rows]

        for result_id in result_ids:
            # Get trained model form DB
            model = load_model_from_db("hourly", result_id)

            # Get features of current model
            model_features = model.feature_names_in_

            # Get features, wich are used for model training
            query = f"""
                SELECT feature_name
                  FROM hourly_rank_forward_selection f
                 WHERE (f.result_id - f.rank) =
                 (
                    SELECT result_id - rank
                    FROM hourly_rank_forward_selection
                    WHERE result_id = {result_id}
                 )
                 ORDER BY f.rank
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            db_features = [row[0] for row in rows]

            if db_features:
                logging.info(f"Result-ID: {result_id}")
                logging.info(f"Model-Features: {model_features}")
                logging.info(f"DB-Features: {db_features}")
                logging.info(f"Difference: {set(model_features).symmetric_difference(set(db_features))}")


if __name__ == "__main__":
    setup_logging()
    main()
