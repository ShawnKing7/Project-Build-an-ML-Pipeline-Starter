import argparse
import json
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import wandb

def go(args):
    with wandb.init() as run:
        # Your training logic here (will add in Step 2)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--rf_config", type=str, default="{}")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    go(args)