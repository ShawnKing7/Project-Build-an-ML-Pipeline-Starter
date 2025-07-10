import argparse
import wandb
import pandas as pd
import sweetviz as sv
import os
import tempfile


def go(args):
    run = wandb.init(job_type="eda")
    run.config.update(args)

    # Download the input artifact
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Load the data
    df = pd.read_csv(artifact_path)

    # Generate Sweetviz report
    report = sv.analyze(df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "eda_report.html")
        report.show_html(report_path)

        # Log the report as an artifact
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description
        )
        artifact.add_file(report_path)
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA with Sweetviz and W&B")
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_description", type=str, required=True)
    args = parser.parse_args()
    go(args)
