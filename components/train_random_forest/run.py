import os
import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import wandb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def go(args):

    run = wandb.init(job_type="train_model")
    run.config.update(vars(args))

    # Load data
    artifact = run.use_artifact(args.trainval_artifact)
    df = pd.read_csv(artifact.file())

    # Split train/val
    stratify = df[args.stratify_by] if args.stratify_by else None
    train, val = train_test_split(
        df,
        test_size=args.val_size,
        stratify=stratify,
        random_state=args.random_seed,
    )

    # Separate target
    X_train = train.drop(columns=["price"])
    y_train = train["price"]
    X_val = val.drop(columns=["price"])
    y_val = val["price"]

    # Load hyperparameters
    import json
    with open(args.rf_config) as f:
        rf_config = json.load(f)

    # Train model
    model = RandomForestRegressor(**rf_config)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    mae = mean_absolute_error(y_val, preds)

    wandb.log({"rmse": rmse, "mae": mae})
    print(f"Validation RMSE: {rmse}")
    print(f"Validation MAE: {mae}")

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, os.path.join("model", "model.joblib"))

    # Export with MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.sklearn.log_model(model, "model")

    # Log to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description="Random Forest model exported as joblib + MLflow"
    )
    artifact.add_file("model/model.joblib")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=False, defaul
