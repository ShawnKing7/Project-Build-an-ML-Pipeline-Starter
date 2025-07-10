import argparse
import wandb

def go(args):
    with wandb.init() as run:
        artifact = wandb.Artifact(
            args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description
        )
        artifact.add_file(args.sample)
        run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--artifact_name", required=True)
    parser.add_argument("--artifact_type", required=True)
    parser.add_argument("--artifact_description", required=True)
    args = parser.parse_args()
    go(args)
