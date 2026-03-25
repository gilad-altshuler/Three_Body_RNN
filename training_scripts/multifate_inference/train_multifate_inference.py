import sys, os
from pathlib import Path

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

import torch
from torch import nn

from methods.models import *
from tasks.MultiFate_task import generate_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs" / "multifate_inference"

def train_multifate_inference(run_name, data_size, N, T, epochs, lr, ranks):

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)

    if (run_dir / "input.pth").exists() and (run_dir / "target.pth").exists():
        input = torch.load(run_dir / "input.pth", map_location=DEVICE, weights_only=True)
        x_half = torch.load(run_dir / "target.pth", map_location=DEVICE, weights_only=True)
        print("✅ Loaded existing multifate data.")
    else:
        # load multifate data 
        params = (dt, Kd, n, alpha, beta, inducers) = (0.2,1,1.5,3.6,90.0,0)
        input, x_half = generate_data(data_size, T, N, *params, DEVICE=DEVICE)
        torch.save(input, run_dir / "input.pth")
        torch.save(x_half, run_dir / "target.pth")

    # set up training parameters
    input_size = output_size = hidden_dim = N
    criterion = torch.nn.MSELoss()
    scheduler = None
    w_out = torch.nn.Identity()
    dataset = (input[:,1:,:], x_half[:,1:,:], x_half[:,0,:])

    print("Training student models...")
    # training student models

    models = [RNN, TBRNN, HORNN]
    for model in models:
        model_name = get_model_str(model)

        if (run_dir / f"{model_name}_student.pth").exists():
            print(f"✅ {model_name} student already trained. Skipping...")
            continue

        print(f"Training {model_name} student...")

        student = model(input_size, output_size, hidden_dim, mode='cont', form='rate',
                        nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid, task="MultiFate_task",
                        noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True, w_out=w_out).to(DEVICE)

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        _ = train(student, dataset, epochs, optimizer, criterion,
                  scheduler=scheduler, batch_size=data_size, clip_gradient=None, keep_best=True, plot=False)

        torch.save(student.state_dict(), run_dir / f"{model_name}_student.pth")

    low_rank_models = [Low_Rank_RNN, Low_Rank_TBRNN, Low_Rank_HORNN]
    for model in low_rank_models:
        model_name = get_model_str(model)

        for rank in range(1, ranks+1):
            if (run_dir / f"r_{rank}_{model_name}_student.pth").exists():
                print(f"✅ {model_name} student with rank {rank} already trained. Skipping...")
                continue

            print(f"Training {model_name} student with rank {rank}...")

            if model == Low_Rank_HORNN:
                student = model(input_size, output_size, hidden_dim, rank_rnn=1, rank_tbrnn=rank,
                                mode='cont', form='rate', nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid,
                                task="MultiFate_task", noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True,
                                w_out=w_out).to(DEVICE)
            else:
                student = model(input_size, output_size, hidden_dim, rank=rank,
                                mode='cont', form='rate', nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid,
                                task="MultiFate_task", noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True,
                                w_out=w_out).to(DEVICE)

            optimizer = torch.optim.Adam(student.parameters(), lr=lr)

            _ = train(student, dataset, epochs, optimizer, criterion,
                      scheduler=scheduler, batch_size=data_size, clip_gradient=None, keep_best=True, plot=False)

            torch.save(student.state_dict(), run_dir / f"r_{rank}_{model_name}_student.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="multifate inference training")
    parser.add_argument("--run_name", type=str, help="Name of the run directory")
    parser.add_argument("--data_size", type=int, default=256, help="Size of the training data")
    parser.add_argument("--N", type=int, default=30, help="Number of proteins in the task")
    parser.add_argument("--T", type=int, default=100, help="Time steps for the task")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of epochs for training the teachers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rates for the teacher models")
    parser.add_argument("--ranks", type=int, default=10, help="Number of ranks to train for low-rank students")
    args, extras = parser.parse_known_intermixed_args()


    train_multifate_inference(run_name=args.run_name, data_size=args.data_size, N=args.N, T=args.T, epochs=args.epochs, lr=args.lr, ranks=args.ranks)