import os
from pathlib import Path
import numpy as np
import h5py
import pickle


ROOT = Path(__file__).absolute().parent.parent.parent.parent
RUN_DIR = ROOT.parent / "ext" / "runs" / "reach_conditioning"
DATA_DIR = ROOT / "data" / "reach_inference" / "reach_condition"
CONFIGS=(
  "r_5_rnn",
  "r_6_rnn",
  "r_4_r_1_hornn",
  "r_5_r_1_hornn",
  "r_6_r_1_hornn",
)
RUNS = 30

if not os.path.isdir(RUN_DIR):
    print(f"Run directory {RUN_DIR} does not exist. Please run the training script first.")
    exit(1)
print("Configs:")
for cfg in CONFIGS: print(cfg)

r2s = {cfg: np.array([h5py.File(RUN_DIR / f"{cfg}/{run:03d}/checkpoints/best/metrics.h5", "r")["vel R2"][()] for run in range(1,RUNS+1)]) for cfg in CONFIGS}

if not os.path.isdir(DATA_DIR):
    DATA_DIR.mkdir(parents=True)

with open(DATA_DIR / "r2s.pkl", "wb") as f:
    pickle.dump(r2s, f)

print("Done. All data saved to:", DATA_DIR)