import argparse
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
import wandb


def go(args):
    run = wandb.init(job_type="train_model")

    # Load data
    df = pd.read_csv(args.input_artifact)
    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed
    )

    # Train model
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed
    )
    rf.fit(X_train, y_train)

    # Log model and parameters
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_seed": args.random_seed
    })
    mlflow.sklearn.log_model(rf, args.output_artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_artifact", type=str, required=True)
    args = parser.parse_args()

    go(args)