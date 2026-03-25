#please run this script onnly after completing training for all tasks
import sys, os
from pathlib import Path
import numpy as np
import torch
import pickle

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from methods.models import Low_Rank_RNN, Low_Rank_HORNN, Low_Rank_TBRNN
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
        "hornn": [[None] * runs for _ in range(ranks)],
        "rnn": [[None] * runs for _ in range(ranks)],
        "tbrnn": [[None] * runs for _ in range(ranks)],
    }
    for m in modes
}

for run in range(1,runs+1):
    i = run-1

    print(f"Reading stats of run: {run:03}")

    train_set = torch.load(RUN_DIR / f"{run:03}" / "train_set.pth", map_location=DEVICE, weights_only=False)
    valid_set = torch.load(RUN_DIR / f"{run:03}" / "valid_set.pth", map_location=DEVICE, weights_only=False)
    test_set = torch.load(RUN_DIR / f"{run:03}" / "test_set.pth", map_location=DEVICE, weights_only=False)

    input,target,hidden,_ = train_set.dataset.tensors
    w_out = torch.nn.Identity()
    output_nonlinearity = (lambda x: x)

    for rank in range(1, ranks+1):

        hornn_student = Low_Rank_HORNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank_rnn=rank,
                                rank_tbrnn=1, task="Mante_task", mode='cont', form='rate',
                                output_nonlinearity=output_nonlinearity, noise_std=0.0, tau=0.2,
                                Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)
        
        if not os.path.exists(path := RUN_DIR / f"{run:03}" / f"r_{rank}_r_1_hornn_student.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        hornn_student.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))

        for m in modes:
            r2s[m]['hornn'][rank-1][i] = evaluate(hornn_student, locals()[f"{m}_set"], r2_mode='per_batch')


        rnn_student = Low_Rank_RNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank=rank,
                               task="Mante_task", mode='cont', form='rate',output_nonlinearity=output_nonlinearity,
                               noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)
        
        # load student models
        if not os.path.exists(path := RUN_DIR / f"{run:03}" / f"r_{rank}_rnn_student.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        rnn_student.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

        # evaluate rnns
        for m in modes:
            r2s[m]['rnn'][rank-1][i] = evaluate(rnn_student, locals()[f"{m}_set"], r2_mode='per_batch')

        tbrnn_student = Low_Rank_TBRNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank=rank,
                               task="Mante_task", mode='cont', form='rate',output_nonlinearity=output_nonlinearity,
                               noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)
        
        # load student models
        if not os.path.exists(path := RUN_DIR / f"{run:03}" / f"r_{rank}_tbrnn_student.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        tbrnn_student.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

        # evaluate tbrnns
        for m in modes:
            r2s[m]['tbrnn'][rank-1][i] = evaluate(tbrnn_student, locals()[f"{m}_set"], r2_mode='per_batch')

models = ['rnn', 'hornn', 'tbrnn']
for model in models:
    for m in modes:
        r2s[m][model] = np.array(r2s[m][model])

if not os.path.isdir(DATA_DIR):
    DATA_DIR.mkdir(parents=True)

with open(DATA_DIR / "r2s.pkl", "wb") as f:
    pickle.dump(r2s, f)

print("Done. All data saved to:", DATA_DIR)