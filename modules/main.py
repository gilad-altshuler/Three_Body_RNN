import torch
import random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import time
from sklearn.decomposition import PCA

import utils
import sin_task
import K_Bit_Flipflop_task
import PIN_task
from Models import *

SEED = 3150
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # k - flip-flop
    K = 1
    T = 100

    # decide on hyperparameters
    input_size = K
    output_size = K
    hidden_dim = 30

    tb_rnn = Three_Bodies_RNN_Dynamics(input_size, output_size, hidden_dim).to(DEVICE)

    # MSE loss and Adam optimizer with a learning rate
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(tb_rnn.parameters(), lr=1e-03)

    # train parameters
    n_steps = 3000
    n_batch = 500

    # generate new data
    Inputs, Targets = K_Bit_Flipflop_task.generate_mem_data(n_batch, T, K)
    losses = tb_rnn.train(Inputs, Targets, n_steps, optimizer, criterion, batch_size=n_batch, T=T)
    print(losses)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
