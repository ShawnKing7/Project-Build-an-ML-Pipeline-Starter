import wandb
import pandas as pd
from pathlib import Path

run = wandb.init(project='nyc_airbnb', job_type='artifact_repair')
sample_path = Path('data/sample.csv')
sample_path.parent.mkdir(exist_ok=True)
pd.DataFrame({'example':[1]}).to_csv(sample_path)
wandb.Artifact('sample', type='raw_data').add_file(str(sample_path)).upload()
run.finish()
