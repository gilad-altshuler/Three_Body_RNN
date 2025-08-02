#please run this script onnly after completing training for all tasks
import sys, os
import importlib
from pathlib import Path
import numpy as np
import torch
import copy
import pickle

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from Models import Low_Rank_RNN, Low_Rank_TBRNN
from tasks.K_Bit_Flipflop_task import generate_data
ROOT = Path(__file__).absolute().parent.parent.parent
RUN_DIR = ROOT.parent / "runs" / "teacher_student"
DATA_DIR = ROOT / "data" / "teacher_student"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

if not os.path.isdir(RUN_DIR):
    print(f"Run directory {RUN_DIR} does not exist. Please run the training script first.")
    exit(1)

hidden_dim = 30
runs = 30
T = 100
data_size = 500
s_ranks = 6

STATS = {
    "rnn": {
        "rnn": [[None for _ in range(runs)] for _ in range(s_ranks)],
        "tbrnn": [[None for _ in range(runs)] for _ in range(s_ranks)]
    },
    "tbrnn": {
        "rnn": [[None for _ in range(runs)] for _ in range(s_ranks)],
        "tbrnn": [[None for _ in range(runs)] for _ in range(s_ranks)]
    }
}

tasks = [("K_Bit_Flipflop_task/1",1),
         ("K_Bit_Flipflop_task/2",2),
         ("K_Bit_Flipflop_task/3",3),
         ("sin_task",2)]

task_stats = []

for task,t_rank in tasks:
    ###################
    if task == 'K_Bit_Flipflop_task/3':
       continue
    ###################
    if not os.path.isdir(RUN_DIR / task):
        print(f"Run directory {RUN_DIR / task} does not exist.")
        continue

    print(f"Task: {task}")
    stats = {}

    if task == 'sin_task':
      input_size = output_size = 1
    else:
      input_size = output_size = t_rank
    hidden_dim = 30

    accs = copy.deepcopy(STATS)
    mse_errs = copy.deepcopy(STATS)
    r2s = copy.deepcopy(STATS)
    t_s_errs = copy.deepcopy(STATS)

    generate_data = getattr(importlib.import_module("tasks."+task.split('/')[0]), 'generate_data')
    evaluate = getattr(importlib.import_module("tasks."+task.split('/')[0]), 'evaluate')

    input, target = generate_data(data_size,T,input_size,DEVICE=DEVICE)
    input = input.to(DEVICE)
    target = target.to(DEVICE)

    for run in range(1,runs+1):
        i = run-1
        print(f"Reading stats of run: {run:03}")
        

        # define teachers
        lr_tbrnn = Low_Rank_TBRNN(input_size, output_size, hidden_dim, rank=t_rank, mode='cont', form='rate', task="FF",
                            noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
        if not os.path.exists(path := RUN_DIR / f"{task}/{run:03}/tbrnn_teacher.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        lr_tbrnn.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

        lr_rnn = Low_Rank_RNN(input_size, output_size, hidden_dim, rank=t_rank, mode='cont', form='rate', task="FF",
                            noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
        if not os.path.exists(path := RUN_DIR / f"{task}/{run:03}/rnn_teacher.pth"):
            print(f"❌ Missing: {path}")
            exit(1)
        lr_rnn.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

        teachers = {"rnn":lr_rnn,"tbrnn":lr_tbrnn}

        for t_name,teacher in teachers.items():
            teacher_hidden = teacher(input,None)[2].detach().clone()

            for s_rank in range(1,s_ranks+1):
            # define students
                lr_rnn_student = Low_Rank_RNN(input_size, output_size, hidden_dim, rank=s_rank,
                                            mode='cont', form='rate',output_nonlinearity=teacher.output_nonlinearity,task="FF",noise_std=0.0,
                                            tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)

                lr_tbrnn_student = Low_Rank_TBRNN(input_size, output_size, hidden_dim, rank=s_rank,
                                            mode='cont', form='rate',output_nonlinearity=teacher.output_nonlinearity,task="FF",noise_std=0.0,
                                            tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)

                students = {"rnn":lr_rnn_student,"tbrnn":lr_tbrnn_student}

                for s_name,student in students.items():
                    if not os.path.exists(path :=RUN_DIR / f"{task}/{run:03}/r_{s_rank}_{s_name}_on_{t_name}.pth"):
                        print(f"❌ Missing: {path}")
                        exit(1)                  

                    student.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

                    (
                    accs[t_name][s_name][s_rank-1][i],
                    mse_errs[t_name][s_name][s_rank-1][i],
                    r2s[t_name][s_name][s_rank-1][i],
                    t_s_errs[t_name][s_name][s_rank-1][i]
                    ) = evaluate(student,input,target,teacher_traj=teacher_hidden,rates=False,r2_mode='per_batch',r2_all=True)

    for t_name in ['rnn','tbrnn']:
      for s_name in ['rnn','tbrnn']:
        accs[t_name][s_name] = np.array(accs[t_name][s_name])
        mse_errs[t_name][s_name] = np.array(mse_errs[t_name][s_name])
        r2s[t_name][s_name] = np.array(r2s[t_name][s_name])
        t_s_errs[t_name][s_name] = np.array(t_s_errs[t_name][s_name])

    task_stats.append({task: {
        "accs": accs,
        "mse_errs": mse_errs,
        "r2s": r2s,
        "t_s_errs": t_s_errs
    }})


with open(DATA_DIR / "task_stats.pkl", "wb") as f:
    pickle.dump(task_stats, f)
print("Done. All data saved to:", DATA_DIR)

