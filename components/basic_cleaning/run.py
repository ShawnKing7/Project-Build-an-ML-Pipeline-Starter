import argparse
import pandas as pd
import wandb

def clean_data(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    # Include version alias like ":latest"
    local_path = run.use_artifact(args.input_artifact + ":latest").file()

    df = pd.read_csv(local_path)

    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    df.dropna(subset=["name", "host_name", "last_review"], inplace=True)

    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_artifact", required=True, type=str)
    parser.add_argument("--output_artifact", required=True, type=str)
    parser.add_argument("--output_type", required=True, type=str)
    parser.add_argument("--output_description", required=True, type=str)
    parser.add_argument("--min_price", required=True, type=float)
    parser.add_argument("--max_price", required=True, type=float)

    args = parser.parse_args()
    clean_data(args)
