#!/usr/bin/env python
import os
import argparse
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):
    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting data into train and temp")
    stratify = df[args.stratify_by] if args.stratify_by else None

    # First split into train and temp (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=args.test_size + args.val_size,
        random_state=args.random_seed,
        stratify=stratify
    )

    # Now split temp into validation and test
    val_prop = args.val_size / (args.test_size + args.val_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - val_prop,
        random_state=args.random_seed
    )

    # Save and log the artifacts
    for split, dataframe in zip(['train', 'val', 'test'], [df_train, df_val, df_test]):
        file_name = f"{split}.csv"
        dataframe.to_csv(file_name, index=False)
        artifact = wandb.Artifact(
            name=f"{split}_data.csv",
            type="dataset",
            description=f"{split} split of data"
        )
        artifact.add_file(file_name)
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")

    parser.add_argument("--input_artifact", type=str, required=True,
                        help="Fully-qualified name for the input CSV artifact")
    parser.add_argument("--test_size", type=float, required=True,
                        help="Fraction of data to reserve for test split")
    parser.add_argument("--val_size", type=float, required=True,
                        help="Fraction of data to reserve for validation split")
    parser.add_argument("--random_seed", type=int, required=True,
                        help="Random seed for reproducibility")
    parser.add_argument("--stratify_by", type=str, required=False, default=None,
                        help="Column to use for stratification (optional)")

    args = parser.parse_args()
    go(args)
