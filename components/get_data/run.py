import argparse
import pandas as pd
import wandb
import os


def go(args):
    # Load local file instead of downloading from internet
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sample.csv"))

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    df = pd.read_csv(local_path)

    # Initialize wandb run
    run = wandb.init(job_type="download_data")
    run.config.update(args)

    # Save temp copy of dataset
    temp_output = "sample.csv"
    df.to_csv(temp_output, index=False)

    # Upload as artifact
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(temp_output)
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and log dataset")
    parser.add_argument("--sample", type=int, help="Not used here, just for compatibility")
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_description", type=str, required=True)

    args = parser.parse_args()

    go(args)
