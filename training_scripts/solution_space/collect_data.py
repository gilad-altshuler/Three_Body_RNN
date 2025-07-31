import sys, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from utils import CKA
from Models import Low_Rank_RNN, Low_Rank_TBRNN, Low_Rank_GRU
from tasks.K_Bit_Flipflop_task import generate_data

ROOT = Path(__file__).absolute().parent.parent.parent
SAVE_RUN = 1
RUN_DIR = ROOT.parent / "runs" / "train_solution_space"
DATA_DIR = ROOT / "data" / "solution_space"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = output_size = K = 1
hidden_dim = 30
runs = 30
matrices = []

model_classes = {
    "RNN": Low_Rank_RNN,
    "TBRNN": Low_Rank_TBRNN,
    "TBRNN_same": Low_Rank_TBRNN,  # same class, different name
    "GRU": Low_Rank_GRU,
}

input = torch.load(DATA_DIR / "input.pth", map_location=DEVICE, weights_only=True)

for name,lr_class in model_classes.items():
    for run in range(1,runs+1):
        lr_model = lr_class(input_size, output_size, hidden_dim, rank=K, mode='cont',
                        noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
        path = RUN_DIR / f"{run:03}/rank_1_{name}.pth"
        if os.path.exists(path):
            lr_model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        else:
            print(f"❌ Missing: {path}")
            exit(1)

        matrices.append(lr_model(input, None)[2].reshape(-1,hidden_dim).cpu().detach().numpy())
print(f"✅ Loaded model matrix of length {len(matrices)}")
corrmatrix = 1.0 - np.array([[CKA(i, j) for j in matrices] for i in matrices])
D = (corrmatrix+corrmatrix.T) / 2


np.save(DATA_DIR / "distance_matrix.npy", D)

print("saving instances of models..")
torch.save(torch.load(RUN_DIR / f"{SAVE_RUN:03d}" / "rank_1_RNN.pth", weights_only=True), DATA_DIR / "rank_1_RNN.pth")
torch.save(torch.load(RUN_DIR / f"{SAVE_RUN:03d}" / "rank_1_TBRNN.pth", weights_only=True), DATA_DIR / "rank_1_TBRNN.pth")
torch.save(torch.load(RUN_DIR / f"{SAVE_RUN:03d}" / "rank_1_GRU.pth", weights_only=True), DATA_DIR / "rank_1_GRU.pth")
print("Done. All data saved to:", DATA_DIR)

