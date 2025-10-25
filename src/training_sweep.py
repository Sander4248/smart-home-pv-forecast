# EXPERIMENTAL

import logging
from pathlib import Path

import yaml

from src.config.logging_config import setup_logging
from src.training_runner import execute_training_runs

ROOT = Path(__file__).resolve().parents[1]


def run_feature_sweep(
        test_base_config_path: Path,
        working_config_path: Path,
        base_features: list[str],
        base_description: str):
    """
    Performs a feature sweep of weather features and starts a training run for each variant.

    The function first reads the base configuration, extracts all weather features defined under 'features.weather', 
    and iterates over them. For each feature (minus those always activated in 'base_features') it overwrites the 
    working configuration (working_config_path) by:
      1) emptying all entries under 'features.weather'
      2) setting the description under 'meta.description' to '{base_description} {feature}'
      3) fills 'features.weather' with 'base_features + [feature]'

    After each write operation, a training run is started for the updated working configuration using the test
    base configuration.

    Side Effects:
      - Overwrites the file under 'working_config_path' with each iteration.
      - Writes log output (depending on logger configuration).
      - Starts a training and, if necessary, evaluation run for each feature variant.

    Preconditions:
      - 'base_config_path' and 'working_config_path' must be valid YAMLs with the keys 'features.weather'
        and 'meta.description'.
      - 'test_base_config_path' must be a valid (test) base configuration for training runs.

    :param base_config_path: Path to the base YAML from which the complete list of available weather features  is read
    :param test_base_config_path: Path to the test base YAML that is passed to as 'base_config_path' to
           'execute_training_runs' during the sweep.
    :param working_config_path: Path to the working YAML, which is overwritten/updated per iteration and then trained
    :param base_features: List of weather features that should be additionally active in each sweep variant
    :param base_description: Basic description of the experiment; the current feature is appended at the end to
           identify the variant
    """
    setup_logging()

    # _daily_base(base_description, test_base_config_path, working_config_path)

    # _hourly_base(base_description, test_base_config_path, working_config_path)

    # _single_sweep(base_description, base_features, test_base_config_path, working_config_path)

    _multi_sweep(base_description, base_features, test_base_config_path, working_config_path)


def _multi_sweep(base_description, base_features, test_base_config_path, working_config_path):
    # Load base YAML to extract the complete list of weather features
    with open(test_base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    weather_features = config["features"]["weather"]

    feature_configs = []

    _search_duplicate_feature_configs(feature_configs)

    for sweep_features in feature_configs:

        base_sweep_features = base_features + sweep_features
        sweep_str = ", ".join(sweep_features)

        # Iterate through all weather features
        for feature in weather_features:

            # Skip features that should always remain active (base features)
            if feature in base_sweep_features:
                continue

            # Load the working YAML configuration
            with open(working_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Clear all entries under "weather"
            config["features"]["weather"].clear()

            # Update experiment description with the current feature
            config["meta"]["description"] = f"{base_description} {sweep_str}, {feature}"

            # Add base features plus the current one
            config["features"]["weather"].extend(base_sweep_features + [feature])

            # Write the modified config back to the working YAML
            with open(working_config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Execute training run with the updated working config
            print(config)
            execute_training_runs([working_config_path], test_base_config_path)


def _search_duplicate_feature_configs(feature_configs):
    # Search duplicate feature configs
    normalized = []
    for config in feature_configs:
        normalized.append((sorted(config), config))
    seen = set()
    duplicates_found = False
    no_duplicates = []
    for config in normalized:
        if tuple(config[0]) in seen:
            logging.critical(f"Duplicate feature config found: {config[1]}")
            duplicates_found = True
        else:
            seen.add(tuple(config[0]))
            no_duplicates.append(config[1])
    if duplicates_found:
        print(no_duplicates)
        exit(0)


def _single_sweep(base_description, base_features, test_base_config_path, working_config_path):
    # Load base YAML to extract the complete list of weather features
    with open(test_base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    weather_features = config["features"]["weather"]
    # Iterate through all weather features
    for feature in weather_features:

        # Skip features that should always remain active (base features)
        if feature in base_features:
            continue

        # Load the working YAML configuration
        with open(working_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Clear all entries under "weather"
        config["features"]["weather"].clear()

        # Update experiment description with the current feature
        config["meta"]["description"] = f"{base_description} {feature}"

        # Add base features plus the current one
        config["features"]["weather"].extend(base_features + [feature])

        # Write the modified config back to the working YAML
        with open(working_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Execute training run with the updated working config
        print(config)
        execute_training_runs([working_config_path], test_base_config_path)


def _hourly_base(base_description, test_base_config_path, working_config_path):
    weather_features = []
    solar_pos_features = []
    timestamp_features = []

    # Clear Sky Feature
    for value in [False, True]:

        # Iterate through all weather features
        for weather_feature in weather_features:

            # Iterate through all solar_pos_features
            for solar_pos_feature in solar_pos_features:

                # Iterate through all solar_pos_features
                for timestamp_feature in timestamp_features:

                    if not value and not weather_feature and not solar_pos_feature and not timestamp_feature:
                        continue

                    # Load the working YAML configuration
                    with open(working_config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)

                    # Clear sky
                    config["training"]["use_clearsky_feature"] = value

                    # Clear all entries
                    config["features"]["weather"].clear()
                    config["features"]["solar_pos"].clear()
                    config["features"]["timestamp"].clear()

                    # Update experiment description with the current feature
                    config["meta"]["description"] = f"{base_description}"

                    # Add features
                    config["features"]["weather"].extend(weather_feature)
                    config["features"]["solar_pos"].extend(solar_pos_feature)
                    config["features"]["timestamp"].extend(timestamp_feature)

                    # Write the modified config back to the working YAML
                    with open(working_config_path, "w", encoding="utf-8") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    print(config)
                    # Execute training run with the updated working config
                    execute_training_runs([working_config_path], test_base_config_path)


def _daily_base(base_description, test_base_config_path, working_config_path):
    # Load base YAML to extract the complete list of weather features
    weather_features = []
    solar_pos_features = []
    timestamp_features = []

    # Clear Sky Feature
    for value in [False, True]:

        # Iterate through all weather features
        for weather_feature in weather_features:

            # Iterate through all solar_pos_features
            for solar_pos_feature in solar_pos_features:

                # Iterate through all solar_pos_features
                for timestamp_feature in timestamp_features:

                    if not value and not weather_feature and not solar_pos_feature and not timestamp_feature:
                        continue

                    # Load the working YAML configuration
                    with open(working_config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)

                    # Clear sky
                    config["training"]["use_clearsky_feature"] = value

                    # Clear all entries
                    config["features"]["weather"].clear()
                    config["features"]["solar_pos"].clear()
                    config["features"]["timestamp"].clear()

                    # Update experiment description with the current feature
                    config["meta"]["description"] = f"{base_description}"

                    # Add features
                    config["features"]["weather"].extend(weather_feature)
                    config["features"]["solar_pos"].extend(solar_pos_feature)
                    config["features"]["timestamp"].extend(timestamp_feature)

                    # Write the modified config back to the working YAML
                    with open(working_config_path, "w", encoding="utf-8") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    print(config)
                    # Execute training run with the updated working config
                    execute_training_runs([working_config_path], test_base_config_path)


if __name__ == "__main__":
    # # Yaml, from which the features are extracted
    # base_config_yaml = r"configs/base.yaml"

    # Basic YAML for sweep training
    test_base_config_yaml = r"configs/test_base.yaml"

    # yaml, which is adjusted in each iteration
    working_config_yaml = r"configs/test.yaml"

    # Weather features that should always be trained
    always_on_features = []

    # Basic description of the test (the current feature is written behind it)
    experiment_base_description = ("Test")

    # base_config_path = (ROOT / base_config_yaml).resolve()
    test_base_config_path = (ROOT / test_base_config_yaml).resolve()
    working_config_path = (ROOT / working_config_yaml).resolve()

    run_feature_sweep(test_base_config_path, working_config_path, always_on_features,
                      experiment_base_description)
