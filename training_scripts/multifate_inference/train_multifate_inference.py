import sys, os
from pathlib import Path

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

import torch
from torch import nn

from Models import *
from tasks.MultiFate_task import generate_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs" / "multifate_inference"

def train_multifate_inference(run_name, data_size, N, T, epochs, lr):

    # load multifate data 
    params = (dt, Kd, n, alpha, beta, inducers) = (0.2,1,1.5,3.6,90.0,0)
    input, x_half = generate_data(data_size, T, N, *params, DEVICE=DEVICE)

    # set up training parameters
    input_size = output_size = hidden_dim = N
    criterion = torch.nn.MSELoss()
    scheduler = None
    hidden=x_half[:,0,:]
    w_out = torch.nn.Identity()

    print("Training student models...")
    # training student models

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)

    # training RNN
    print("Training RNN student...")
    rnn_student = RNN(input_size, output_size, hidden_dim, mode='cont', form='rate',
                      nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid, task="MultiFate_task",
                      noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True, w_out=w_out).to(DEVICE)

    optimizer = torch.optim.Adam(rnn_student.parameters(), lr=lr)

    _ = train(rnn_student,input[:,1:,:],x_half[:,1:,:],epochs,optimizer,criterion,
              scheduler=scheduler,mask_train=None,batch_size=data_size,
              hidden=hidden,clip_gradient=None,keep_best=True,plot=False)

    # training TBRNN
    print("Training TBRNN student...")
    tbrnn_student = TBRNN(input_size, output_size, hidden_dim, mode='cont', form='rate',
                          nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid, task="MultiFate_task",
                          noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True, w_out=w_out).to(DEVICE)

    optimizer = torch.optim.Adam(tbrnn_student.parameters(), lr=lr)
    _ = train(tbrnn_student,input[:,1:,:],x_half[:,1:,:],epochs,optimizer,criterion,
              scheduler=scheduler,mask_train=None,batch_size=data_size,
              hidden=hidden,clip_gradient=None,keep_best=True,plot=False)

    # training HORNN
    print("Training HORNN student...")
    hornn_student = HORNN(input_size, output_size, hidden_dim, mode='cont', form='rate',
                          nonlinearity=torch.tanh, output_nonlinearity=torch.sigmoid, task="MultiFate_task",
                          noise_std=0.0, tau=0.2, Win_bias=True, Wout_bias=True, w_out=w_out).to(DEVICE)

    optimizer = torch.optim.Adam(hornn_student.parameters(), lr=lr)

    _ = train(hornn_student,input[:,1:,:],x_half[:,1:,:],epochs,optimizer,criterion,
              scheduler=scheduler,mask_train=None,batch_size=data_size,
              hidden=hidden,clip_gradient=None,keep_best=True,plot=False)

    torch.save(input, run_dir / "input.pth")
    torch.save(x_half, run_dir / "target.pth")
    torch.save(rnn_student.state_dict(), run_dir / "RNN_student.pth")
    torch.save(tbrnn_student.state_dict(), run_dir / "TBRNN_student.pth")
    torch.save(hornn_student.state_dict(), run_dir / "HORNN_student.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="multifate inference training")
    parser.add_argument("--run_name", type=str, help="Name of the run directory")
    parser.add_argument("--data_size", type=int, default=256, help="Size of the training data")
    parser.add_argument("--N", type=int, default=30, help="Number of proteins in the task")
    parser.add_argument("--T", type=int, default=100, help="Time steps for the task")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of epochs for training the teachers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rates for the teacher models")
    args, extras = parser.parse_known_intermixed_args()


    train_multifate_inference(run_name=args.run_name, data_size=args.data_size, N=args.N, T=args.T, epochs=args.epochs, lr=args.lr)