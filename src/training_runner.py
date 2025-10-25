from pathlib import Path

from src.config.logging_config import setup_logging
from src.config.yaml_loader import load_run_cfgs
from src.training.training_pipeline import run_training_pipeline
from src.forecast.auto_evaluation import run_auto_evaluate


def execute_training_runs(
        config_file_paths: list[str | Path],
        base_config_path: str | Path = None,
        description: str = None
):
    """
    Executes one or more training runs based on YAML configurations.

    This function initializes logging, loads all transferred configuration files one after the other (optionally
    together with a base configuration) and starts the training pipeline for each loaded run configuration.
    Optionally, an automatic evaluation is then performed for each run, provided this is enabled in the respective
    configuration (cfg.evaluation.auto_eval).

    Side Effects:
      - Writes log output (depending on the logger configuration).
      - Starts training runs (e.g., training, checkpoints, artifacts).
      - Starts automatic evaluations after training, if applicable.

    :param config_file_paths: List of paths to YAML configuration files.
           Each entry describes a training run (or refers to a run derived from a base configuration).
    :param base_config_path: Optional path to a base YAML.
           If set, this is used as the basis when loading the run configurations.
    :param description: Optional description of the training run.
    :return:
    """
    setup_logging()

    if base_config_path:
        # Load configurations using an explicit base config
        cfgs = load_run_cfgs(config_file_paths, base_config_path)
    else:
        # If no base_config_path is given, load_run_cfgs will use its default
        cfgs = load_run_cfgs(config_file_paths)


    # Run training for each configuration
    for cfg in cfgs:

        # Set optional description
        if description:
            cfg.meta.description = description

        cfg.evaluation.eval_with_weather_history_data = True
        run_training_pipeline(cfg)

        # Automatically run evaluation if enabled in config
        if cfg.evaluation.auto_eval:

            # print(cfg.meta.description)
            run_auto_evaluate(cfg)


if __name__ == "__main__":
    # Example execution with a test config and a base config
    config_file_paths = [
        "configs/test.yaml",
    ]

    base_config_path = "configs/base.yaml"
    description = None
    execute_training_runs(config_file_paths, base_config_path, description)
