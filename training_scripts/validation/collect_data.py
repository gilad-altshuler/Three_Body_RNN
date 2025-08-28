import numpy as np, os
import torch
from pathlib import Path

TASK = "K_Bit_Flipflop_task"
ROOT = Path(__file__).absolute().parent.parent.parent
RUN = 1
RUN_DIR = ROOT.parent / "runs" / "low_rank_procedures" / TASK
DATA_DIR = ROOT / "data" / "validation"
lr_methods = ["tca", "tt", "slice_tca", "lint"]
runs = 30

for m in lr_methods:
    data = []
    for run in range(1, runs+1):
        path = RUN_DIR / f"{run:03d}/{m}.npy"
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape == (6,):
                data.append(arr)
            else:
                print(f"⚠️ Skipping {path}: shape {arr.shape} != (6,)")
        else:
            print(f"❌ Missing: {path}")

    stacked = np.stack(data)
    print(f"✅ {m} loaded shape:", stacked.shape)
    np.save(DATA_DIR / f"all_{m}.npy", stacked)

print("saving teacher tbrnn model..")
torch.save(torch.load(RUN_DIR / f"{RUN:03d}" / "teacher_model.pth", weights_only=True), DATA_DIR / "teacher_model.pth")

print("Done. All data saved to:", DATA_DIR)

