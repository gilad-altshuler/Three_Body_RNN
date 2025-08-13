import sys, os
from pathlib import Path

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

import torch

from Models import *
from tasks.Mante_task import generate_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs" / "mante_inference"
DATA_ROOT = Path(__file__).absolute().parent.parent.parent / "data" / "mante_inference"

def train_mante_inference(run_name, ranks, epochs, batch_size, lr):

    # load mante data
    train_dataset, valid_dataset, test_dataset = generate_data(DATA_ROOT, DEVICE=DEVICE)
    input,target,hidden,_ = train_dataset.dataset.tensors

    criterion = torch.nn.MSELoss()

    w_out = torch.nn.Identity()
    output_nonlinearity = (lambda x: x)

    print("Training student models...")
    # training student models

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)

    print(f"Training low rank hornn student...")

    student = Low_Rank_HORNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank_rnn=1,
                             rank_tbrnn=1, task="Mante_task", mode='cont', form='rate',
                             output_nonlinearity=output_nonlinearity, noise_std=0.0, tau=0.2,
                             Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr[0])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=lr[1]/lr[0],total_iters=epochs)

    _ = train(student, train_dataset, epochs, optimizer, criterion,
                valid_set=valid_dataset, scheduler=scheduler, batch_size=batch_size,
                clip_gradient=1, keep_best=True, plot=False)

    torch.save(student.state_dict(), run_dir / f"r_1_r_1_hornn_student.pth")

    for rank in range(1, ranks+1):
        print(f"Training rank-{rank} rnn student...")

        student = Low_Rank_RNN(input.shape[-1], target.shape[-1],hidden.shape[-1], rank=rank,
                               task="Mante_task", mode='cont', form='rate',output_nonlinearity=output_nonlinearity,
                               noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False, w_out=w_out).to(DEVICE)

        optimizer = torch.optim.Adam(student.parameters(), lr=lr[0])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=lr[1]/lr[0],total_iters=epochs)

        _ = train(student, train_dataset, epochs, optimizer, criterion,
                  valid_set=valid_dataset, scheduler=scheduler, batch_size=batch_size,
                  clip_gradient=1, keep_best=True, plot=False)

        torch.save(student.state_dict(), run_dir / f"r_{rank}_rnn_student.pth")

    torch.save(train_dataset, run_dir / "train_set.pth")
    torch.save(valid_dataset, run_dir / "valid_set.pth")
    torch.save(test_dataset, run_dir / "test_set.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="multifate inference training")
    parser.add_argument("--run_name", type=str, help="Name of the run directory")
    parser.add_argument("--ranks", type=int, default=5, help="Number ranks to train rnn student")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for training the students")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, nargs=2, default=[5e-2, 1e-3], help="Learning rates for the student models")
    args, extras = parser.parse_known_intermixed_args()


    train_mante_inference(run_name=args.run_name, ranks=args.ranks, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)