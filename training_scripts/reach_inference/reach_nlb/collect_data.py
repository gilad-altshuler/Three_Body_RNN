import os
from pathlib import Path
import numpy as np
import h5py
import pickle


ROOT = Path(__file__).absolute().parent.parent.parent
RUN_DIR = ROOT.parent / "ext" / "runs" / "reach_nlb"
DATA_DIR = ROOT / "data" / "reach_inference" / "reach_nlb"
CONFIGS=(
  "36,rnn"
  "34,1,hornn"
  "35,1,hornn"
  "36,1,hornn"
)
RUNS = 30
METRICS = ["co-bps","fp-bps","psth R2","vel R2"]

if not os.path.isdir(RUN_DIR):
    print(f"Run directory {RUN_DIR} does not exist. Please run the training script first.")
    exit(1)
print("Configs:")
for cfg in CONFIGS: print(cfg)

metrics = {metric: {cfg: np.array([h5py.File(RUN_DIR / f"{cfg}/{run:03d}/checkpoints/best/metrics.h5", "r")[metric][()] for run in range(1,RUNS)]) for cfg in CONFIGS} for metric in METRICS}

if not os.path.isdir(DATA_DIR):
    DATA_DIR.mkdir(parents=True)

with open(DATA_DIR / "metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("Done. All data saved to:", DATA_DIR)


