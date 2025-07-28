import numpy as np, os

base = "runs/low_rank_procedures"

methods = ["TCA", "TT", "Slice-TCA", "LINT"]

for m in methods:
    data = []
    for i in range(1, 33):
        path = f"{base}/{i:03d}/{m}.npy"
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
    np.save(f"{base}/all_{m}.npy", stacked)
