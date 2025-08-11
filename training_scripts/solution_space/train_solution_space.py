import sys, os
from pathlib import Path
sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

import torch
from torch import nn

from tasks.K_Bit_Flipflop_task import generate_data
from Models import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs" / "train_solution_space"

def train_solution_space(run_name, rank=1, data_size=128, hidden_dim=30, input_size=1, output_size=1, T=100, epochs=20000):

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)

    input,target = generate_data(data_size, T, input_size, DEVICE=DEVICE)
    dataset = (input, target)

    lr_rnn = Low_Rank_RNN(input_size, output_size, hidden_dim, rank=rank, mode='cont', form='rate', task="K_Bit_Flipflop_task",
                            noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
    lr_tbrnn = Low_Rank_TBRNN(input_size, output_size, hidden_dim, rank=rank, mode='cont', form='rate', task="K_Bit_Flipflop_task",
                            noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
    lr_tbrnn_same = Low_Rank_TBRNN(input_size, output_size, hidden_dim, rank=rank, mode='cont', form='rate', task="K_Bit_Flipflop_task",
                            noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
    lr_gru = Low_Rank_GRU(input_size, output_size, hidden_dim, rank=rank, mode='cont', task="K_Bit_Flipflop_task",
                            noise_std=0.0, tau=0.2, Win_bias=False, Wout_bias=False).to(DEVICE)
    # MSE loss and Adam optimizer with a learning rate
    criterion = nn.MSELoss().to(DEVICE)

    ############### train rnn ###############
    optimizer = torch.optim.Adam(lr_rnn.parameters(), lr=1e-02)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.5,total_iters=epochs)

    _ = train(lr_rnn,dataset,epochs,optimizer,criterion,
              scheduler=scheduler,batch_size=data_size,clip_gradient=None,keep_best=True,plot=False)

    ############### train tbrnn with warmup ###############
    lr_tbrnn.w_in = copy.deepcopy(lr_rnn.w_in)
    lr_tbrnn.w_out = copy.deepcopy(lr_rnn.w_out)
    lr_tbrnn.w_in.weight.requires_grad = False
    lr_tbrnn.w_out.weight.requires_grad = False
    
    optimizer = torch.optim.Adam(lr_tbrnn.parameters(), lr=5e-03)
    _ = train(lr_tbrnn,dataset,epochs//2,optimizer,criterion,
              scheduler=None,batch_size=data_size,clip_gradient=None,keep_best=True,plot=False)
    
    # changing so the w_in and w_out are trainable
    lr_tbrnn.w_in.weight.requires_grad = True
    lr_tbrnn.w_out.weight.requires_grad = True

    # continue training tbrnn
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.5,total_iters=epochs)
    _ = train(lr_tbrnn,dataset,epochs//2,optimizer,criterion,
              scheduler=None,batch_size=data_size,clip_gradient=None,keep_best=True,plot=False)

    ############### train tbrnn with same w_in and w_out ###############

    optimizer = torch.optim.Adam(lr_tbrnn_same.parameters(), lr=5e-03)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.5,total_iters=epochs)

    lr_tbrnn_same.w_in = copy.deepcopy(lr_rnn.w_in)
    lr_tbrnn_same.w_out = copy.deepcopy(lr_rnn.w_out)
    lr_tbrnn_same.w_in.weight.requires_grad = False
    lr_tbrnn_same.w_out.weight.requires_grad = False

    _ = train(lr_tbrnn_same,dataset,epochs,optimizer,criterion,
              scheduler=None,batch_size=data_size,clip_gradient=None,keep_best=True,plot=False)

    ############### train gru ###############

    optimizer = torch.optim.Adam(lr_gru.parameters(), lr=1e-02)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.5,total_iters=epochs)

    _ = train(lr_gru,dataset,epochs,optimizer,criterion,
              scheduler=None,batch_size=data_size,clip_gradient=None,keep_best=True,plot=False)

    # train tbrnn
    torch.save(lr_rnn.state_dict(), run_dir / f"rank_{rank}_RNN.pth")
    torch.save(lr_tbrnn.state_dict(), run_dir / f"rank_{rank}_TBRNN.pth")
    torch.save(lr_tbrnn_same.state_dict(), run_dir / f"rank_{rank}_TBRNN_same.pth")
    torch.save(lr_gru.state_dict(), run_dir / f"rank_{rank}_GRU.pth")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="solution space training")
    parser.add_argument("--run_name", type=str, help="Name of the run directory")
    parser.add_argument("--rank", type=int, default=1, help="Rank of the low-rank training")
    parser.add_argument("--data_size", type=int, default=128, help="Number of samples to generate")
    parser.add_argument("--hidden_dim", type=int, default=30, help="Hidden dimension of the model")
    parser.add_argument("--input_size", type=int, default=1, help="Input size of the model")
    parser.add_argument("--output_size", type=int, default=1, help="Output size of the model")
    parser.add_argument("--T", type=int, default=100, help="Number of time steps")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of training epochs")
    args, extras = parser.parse_known_intermixed_args()


    train_solution_space(run_name=args.run_name, rank=args.rank, data_size=args.data_size, hidden_dim=args.hidden_dim,
                          input_size=args.input_size, output_size=args.output_size, T=args.T, epochs=args.epochs)
