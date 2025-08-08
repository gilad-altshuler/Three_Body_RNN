#please run this script onnly after completing training for all tasks
import sys, os
from pathlib import Path
import numpy as np
import torch
import pickle

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from Models import RNN, TBRNN, HORNN, get_model_str
from utils import heavy_cka,cka_linear_streaming
ROOT = Path(__file__).absolute().parent.parent.parent
RUN_DIR = ROOT.parent / "runs" / "multifate_inference"
DATA_DIR = ROOT / "data" / "multifate_inference"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE="cpu"
if not os.path.isdir(RUN_DIR):
    print(f"Run directory {RUN_DIR} does not exist. Please run the training script first.")
    exit(1)

data_size = 256
T = 100
input_size = output_size = hidden_dim = N = 30

runs = 30

models = [RNN, TBRNN, HORNN]

stats = {
    "rnn": [None for _ in range(runs)],
    "tbrnn": [None for _ in range(runs)],
    "hornn": [None for _ in range(runs)],
}

for run in range(1,runs+1):
    i = run-1
    print(f"Reading stats of run: {run:03}")

    input = torch.load(RUN_DIR / f"{run:03}" / "input.pth", map_location=DEVICE, weights_only=True)
    x_half = torch.load(RUN_DIR / f"{run:03}" / "target.pth", map_location=DEVICE, weights_only=True)
    
    for model in models:
        # define teachers
        student = model(input_size, output_size, hidden_dim, mode='cont', form='rate',
                        nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid, task="MultiFate_task",
                        noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True, w_out=torch.nn.Identity()).to(DEVICE)
        # load student models
        model_name = get_model_str(model)
        if not os.path.exists(path := RUN_DIR / f"{run:03}" / f"{model_name}_student.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        student.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

        # evaluate students
        multifate = x_half[:,1:].reshape(-1,hidden_dim)
        trajectory = student(input[:,1:,:],x_half[:,0,:])[0].reshape(-1,hidden_dim).detach().clone()
        cka_score = cka_linear_streaming(multifate, trajectory)
        torch.cuda.empty_cache()

        stats[model_name][i] = cka_score.item()

for model in models:
    model_name = get_model_str(model)
    stats[model_name] = np.array(stats[model_name])

if not os.path.isdir(DATA_DIR):
    DATA_DIR.mkdir(parents=True)

with open(DATA_DIR / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)
print("Done. All data saved to:", DATA_DIR)