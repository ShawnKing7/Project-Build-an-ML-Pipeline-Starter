#!/usr/bin/env python
"""
Check for data drift using KL divergence between current and reference datasets.
"""

import argparse
import logging
import pandas as pd
import scipy.stats
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def kl_divergence(p, q):
    """Safely compute KL divergence for discrete distributions"""
    p = p + 1e-10
    q = q + 1e-10
    return scipy.stats.entropy(p, q)


def go(args):
    run = wandb.init(job_type="data_check")

    # Download current and reference datasets
    artifact_csv = run.use_artifact(args.csv)
    artifact_ref = run.use_artifact(args.ref)

    current_path = artifact_csv.file()
    reference_path = artifact_ref.file()

    current = pd.read_csv(current_path)
    reference = pd.read_csv(reference_path)

    common_cols = set(current.columns).intersection(reference.columns)

    for col in common_cols:
        if current[col].dtype == object:
            continue  # Only compare numeric columns

        logger.info(f"Checking drift in column: {col}")
        p = current[col].value_counts(normalize=True, bins=10, sort=False)
        q = reference[col].value_counts(normalize=True, bins=10, sort=False)

        score = kl_divergence(p, q)
        logger.info(f"KL divergence for {col}: {score:.5f}")

        if score > args.kl_threshold:
            raise ValueError(f"KL divergence for {col} is too high: {score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data drift checker")

    parser.add_argument("--csv", type=str, required=True, help="Fully-qualified name for cleaned dataset artifact")
    parser.add_argument("--ref", type=str, required=True, help="Fully-qualified name for reference dataset artifact")
    parser.add_argument("--kl_threshold", type=float, required=True, help="Maximum allowed KL divergence")

    args = parser.parse_args()
    go(args)
