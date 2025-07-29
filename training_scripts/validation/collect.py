import numpy as np, os
import torch
from pathlib import Path

TASK = "K_Bit_Flipflop_task"
ROOT = Path(__file__).absolute().parent.parent.parent
RUN = 1
RUN_DIR = ROOT.parent / "runs" / "low_rank_procedures" / TASK
SAVE_DIR = ROOT / "Three_Body_RNN" / "data" / "validation"
methods = ["TCA", "TT", "Slice-TCA", "LINT"]

for m in methods:
    data = []
    for i in range(1, 33):
        path = RUN_DIR / f"{i:03d}/{m}.npy"
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape == (6,):
                data.append(arr)
            else:
                print(f"⚠️ Skipping {path}: shape {arr.shape} != (6,)")
        else:
            print(f"❌ Missing: {path}")

    stacked = np.stack(data)
    print("✅ Loaded shape:", stacked.shape)
    np.save(SAVE_DIR / f"all_{m}.npy", stacked)

    print("saving teacher tbrnn model..")
    torch.save(torch.load(RUN_DIR / f"{RUN:03d}" / "tbrnn_teacher.pth"), SAVE_DIR / f"teacher_model.pth")

    print("Done. All data saved to:", SAVE_DIR)

