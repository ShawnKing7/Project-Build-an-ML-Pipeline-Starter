import argparse
import pandas as pd
import joblib
import wandb
from sklearn.metrics import mean_squared_error
import os

def go(args):
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    # Load test data
    test_path = wandb.use_artifact(args.test_data).file()
    df_test = pd.read_csv(test_path)

    X_test = df_test.drop(columns=[args.target])
    y_test = df_test[args.target]

    # Load model
    model_path = wandb.use_artifact(args.model_export).download()
    model_file = os.path.join(model_path, "model.joblib")
    model = joblib.load(model_file)

    # Predict + evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Test RMSE: {rmse}")
    wandb.log({"test_rmse": rmse})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_export", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()
    go(args)
