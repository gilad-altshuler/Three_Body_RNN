#please run this script onnly after completing training for all tasks
import sys, os
from pathlib import Path
import numpy as np
import torch
import copy
import pickle

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from Models import Low_Rank_RNN, Low_Rank_HORNN
from tasks.Mante_task import evaluate

ROOT = Path(__file__).absolute().parent.parent.parent
RUN_DIR = ROOT.parent / "runs" / "mante_inference"
DATA_DIR = ROOT / "data" / "mante_inference"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isdir(RUN_DIR):
    print(f"Run directory {RUN_DIR} does not exist. Please run the training script first.")
    exit(1)

runs = 30
ranks = 5

modes = ["train", "valid", "test"]

r2s = {
    m: {
        "hornn": [None] * runs,
        "rnn": [[None] * runs for _ in range(ranks)]
    }
    for m in modes
}

per_neuron_r2s = copy.deepcopy(r2s)

for run in range(1,runs+1):
    i = run-1
    if run==25:
        continue
    if run>25:
        i=i-1
    print(f"Reading stats of run: {run:03}")

    train_set = torch.load(RUN_DIR / f"{run:03}" / "train_set.pth", map_location=DEVICE, weights_only=False)
    valid_set = torch.load(RUN_DIR / f"{run:03}" / "valid_set.pth", map_location=DEVICE, weights_only=False)
    test_set = torch.load(RUN_DIR / f"{run:03}" / "test_set.pth", map_location=DEVICE, weights_only=False)

    input,target,hidden,_ = train_set.dataset.tensors
    w_out = torch.nn.Identity()
    output_nonlinearity = (lambda x: x)

    student = Low_Rank_HORNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank_rnn=1,
                             rank_tbrnn=1, task="Mante_task", mode='cont', form='rate',
                             output_nonlinearity=output_nonlinearity, noise_std=0.0, tau=0.2,
                             Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)
    
    if not os.path.exists(path := RUN_DIR / f"{run:03}" / "r_1_r_1_hornn_student.pth"):
        print(f"❌ Missing: {path}")
        exit(1)
    student.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))

    for m in modes:
        r2s[m]['hornn'][i] = evaluate(student, locals()[f"{m}_set"], r2_mode='per_batch')
        per_neuron_r2s[m]['hornn'][i] = evaluate(student, locals()[f"{m}_set"], r2_mode='per_neuron')

    for rank in range(1, ranks+1):
        # define teachers
        student = Low_Rank_RNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank=rank,
                               task="Mante_task", mode='cont', form='rate',output_nonlinearity=output_nonlinearity,
                               noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)
        
        # load student models
        if not os.path.exists(path := RUN_DIR / f"{run:03}" / f"r_{rank}_rnn_student.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        student.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

        # evaluate rnns
        for m in modes:
            r2s[m]['rnn'][rank-1][i] = evaluate(student, locals()[f"{m}_set"], r2_mode='per_batch')
            per_neuron_r2s[m]['rnn'][rank-1][i] = evaluate(student, locals()[f"{m}_set"], r2_mode='per_neuron')

models = ['rnn','hornn']
for model in models:
    for m in modes:
        r2s[m][model] = np.array(r2s[m][model])
        per_neuron_r2s[m][model] = np.array(per_neuron_r2s[m][model])


if not os.path.isdir(DATA_DIR):
    DATA_DIR.mkdir(parents=True)

with open(DATA_DIR / "r2s.pkl", "wb") as f:
    pickle.dump(r2s, f)

with open(DATA_DIR / "per_neuron_r2s.pkl", "wb") as f:
    pickle.dump(per_neuron_r2s, f)

print("Done. All data saved to:", DATA_DIR)