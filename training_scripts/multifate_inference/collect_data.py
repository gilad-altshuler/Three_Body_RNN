#please run this script onnly after completing training for all tasks
import sys, os
from pathlib import Path
import numpy as np
import torch
import pickle

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from methods.models import RNN, TBRNN, HORNN, get_model_str
from tasks.MultiFate_task import evaluate

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
    "cka": {
    "rnn": [None for _ in range(runs)],
    "tbrnn": [None for _ in range(runs)],
    "hornn": [None for _ in range(runs)],
    },
    "r2": {
    "rnn": [None for _ in range(runs)],
    "tbrnn": [None for _ in range(runs)],
    "hornn": [None for _ in range(runs)],
    },
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
        r2, cka = evaluate(student, input[:,1:,:], x_half[:,1:], hidden=x_half[:,0,:], r2_mode='per_neuron', r2_all=False)
        stats['r2'][model_name][i] = r2
        stats['cka'][model_name][i] = cka


for model in models:
    model_name = get_model_str(model)
    stats['r2'][model_name] = np.array(stats['r2'][model_name])
    stats['cka'][model_name] = np.array(stats['cka'][model_name])

if not os.path.isdir(DATA_DIR):
    DATA_DIR.mkdir(parents=True)

with open(DATA_DIR / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)
print("Done. All data saved to:", DATA_DIR)