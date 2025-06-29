import os
import json
import tempfile

import hydra
from omegaconf import DictConfig
import mlflow
import wandb


_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]


@hydra.main(config_name="config", config_path=".")
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_param = config["main"]["steps"]
    active_steps = steps_param.split(",") if steps_param != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw_file_as_downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",
                "main",
                parameters={
                    "input_artifact": "sample.csv",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Data with outliers and nulls removed",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                parameters={
                    "csv": "clean_sample.csv",
                    "ref": "clean_sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"]
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/data_split",
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv",
                    "test_size": config["modeling"]["test_size"],
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:
            rf_config_path = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config_path, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"]), fp)

            mlflow.run(
                f"{config['main']['components_repository']}/train_random_forest",
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config_path,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                },
            )


if __name__ == "__main__":
    go()