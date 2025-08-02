import sys, os
from pathlib import Path
import importlib
sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

import torch
from torch import nn

from Models import *
from low_rank_methods import LINT_method

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs" / "teacher_student"

def train_teacher_student(run_name, t_rank, ranks=[1,6], data_size=128, hidden_dim=30, input_size=1,
                          output_size=1, T=100, epochs=20000, lint_epochs=40000, sched_epochs=20000,
                          lr=[1e-3,5e-4], lint_lr=[1e-2,5e-3], rates=False, w_out_bias=True):

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)

    generate_data = getattr(importlib.import_module("tasks."+run_name.split('/')[0]), 'generate_data')
    input,target = generate_data(data_size, T, input_size, DEVICE=DEVICE)

    criterion = nn.MSELoss().to(DEVICE)
    end_factor = lr[1] / lr[0]
    eps = 0.005
    # training teacher models
    print("Training teacher models...")

    models = [Low_Rank_RNN, Low_Rank_TBRNN]

    teachers = [None, None]

    for i, model in enumerate(models):
        name = get_model_str(model)
        print(f"Training {name} teacher...")
        # training rnn
        while True:
            lr_model = model(input_size, output_size, hidden_dim, rank=t_rank, mode='cont', form='rate', task="Sin",
                              noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=w_out_bias).to(DEVICE)
            
            if (run_dir / f"{name}_teacher.pth").exists():
                # If the teacher model already exists, load it
                print(f"{name} teacher already exists, loading...")
                lr_model.load_state_dict(torch.load(run_dir / f"{name}_teacher.pth"))
                teachers[i] = lr_model
                break

            optimizer = torch.optim.Adam(lr_model.parameters(), lr=lr[0])
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=epochs)
            last_loss = train(lr_model, input, target, epochs, optimizer, criterion,
                              scheduler=scheduler, mask_train=None, batch_size=data_size,
                              hidden=None, clip_gradient=None, keep_best=True, plot=False)[-1]
            if last_loss <= eps:
                teachers[i] = lr_model
                torch.save(lr_model.state_dict(), run_dir / f"{name}_teacher.pth")
                print(f"{name} teacher trained successfully.")
                break
            else:
                print(f"{name} teacher did not converge, last loss: {last_loss}")
                print("Retrying with a new model...")

    print("Teachers trained successfully.")

    # training student models
    print("Training student models with LINT...")
    for teacher in teachers:
        for student_class in models:
            print(f"Training {get_model_str(type(teacher))} with {get_model_str(student_class)}")
            _ = LINT_method(teacher, student_class, input, target, start_rank=ranks[0], end_rank=ranks[1],
                            epochs=lint_epochs, batch_size=data_size, lr=lint_lr, to_save=run_dir, return_accs=False,
                            rates=rates, sched_epochs=sched_epochs)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="solution space training")
    parser.add_argument("--run_name", type=str, help="Name of the run directory")
    parser.add_argument("--t_rank", type=int, default=1, help="Rank of the teacher model")
    parser.add_argument("--ranks", type=int, nargs=2, default=[1, 6], help="Range of ranks for the student models")
    parser.add_argument("--data_size", type=int, default=128, help="Size of the training data")
    parser.add_argument("--hidden_dim", type=int, default=30, help="Hidden dimension of the models")
    parser.add_argument("--input_size", type=int, default=1, help="Input size of the models")
    parser.add_argument("--output_size", type=int, default=1, help="Output size of the models")
    parser.add_argument("--T", type=int, default=100, help="Time steps for the task")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs for training the teachers")
    parser.add_argument("--lint_epochs", type=int, default=40000, help="Number of epochs for LINT training")
    parser.add_argument("--sched_epochs", type=int, default=20000, help="Number of epochs for the LINT lr scheduler")
    parser.add_argument("--lr", type=float, nargs=2, default=[1e-3, 5e-4], help="Learning rates for the teacher models")
    parser.add_argument("--lint_lr", type=float, nargs=2, default=[1e-2, 5e-3], help="Learning rates for the LINT method")
    parser.add_argument("--rates", action='store_true', help="Use rates instead of states for training")
    parser.add_argument("--w_out_bias", action='store_true', help="Whether to use output bias in the models")
    args, extras = parser.parse_known_intermixed_args()


    train_teacher_student(run_name=args.run_name, t_rank=args.t_rank, ranks=args.ranks, data_size=args.data_size, 
                          hidden_dim=args.hidden_dim, input_size=args.input_size, output_size=args.output_size, T=args.T,
                          epochs=args.epochs, lint_epochs=args.lint_epochs, sched_epochs=args.sched_epochs, lr=args.lr,
                          lint_lr=args.lint_lr, rates=args.rates, w_out_bias=args.w_out_bias)




