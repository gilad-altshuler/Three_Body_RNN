"""
Code is based on:
Adrian Valente, May 2022.

Train low-rank networks on the Mante data.
Low-rank networks are pre-initialized with connectivity from a previously trained full-rank.
"""

from low_rank_rnns import mante
import pandas as pd
import torch
from torch.utils.data import random_split, TensorDataset
import numpy as np

# from low_rank_rnns import mante, stats, data_loader_mante as dlm, helpers

# smoothing_width = 50

# hidden_neurons = 0
# n_epochs = 2
# lr = 5e-4
# load_modnum = 22  # Numbering of the full-rank fitted network
def generate_data(DATA_DIR, monkey = 'A', bin_width = 5, DEVICE="cpu"):
    assert monkey in ['A', 'F'], "Monkey must be 'A' or 'F'"

    modnum = 24  # Number of the low-rank fitted networks

    # Load preprocessed condition-averaged data (3d tensor, see 2112_mante_monkey_fits.ipynb)
    conditions = pd.read_csv(DATA_DIR / f'conditions_monkey{monkey}.csv')
    X = np.load(DATA_DIR / f'X_cent_monkey{monkey}.npy')
    nconds, ntime, n_neurons = X.shape
    print(f"Training on monkey {monkey}, version {modnum}")
    print(X.shape)

    # Prepare pseudo-inputs
    ## NEW SETUP: the fitted network is let free for the first 350 ms (while receiving contextual signals).
    mante.fixation_duration = 0
    mante.ctx_only_pre_duration = 350
    mante.stimulus_duration = 650   # counting first step
    mante.delay_duration = 80
    mante.decision_duration = 20


    mante.deltaT = bin_width
    mante.SCALE = 1
    mante.SCALE_CTX = 1
    mante.setup()
    correct_trials = conditions.correct == 1
    input, _, _ = mante.generate_mante_data_from_conditions(conditions[correct_trials]['stim_dir'].to_numpy(),
                                                            conditions[correct_trials]['stim_col'].to_numpy(),
                                                            conditions[correct_trials]['context'].to_numpy())
    target = np.concatenate([np.zeros((nconds, mante.ctx_only_pre_duration_discrete, n_neurons)), X], axis=1)
    mask = torch.ones((nconds, mante.ctx_only_pre_duration_discrete + ntime, n_neurons)).to(DEVICE,dtype=torch.bool)
    mask[:, :mante.ctx_only_pre_duration_discrete, :] = False

    # Prepare training set, initial states for trajectories...
    n_train = int(0.8 * nconds)
    n_valid = int(0.1 * nconds)
    n_test = nconds - n_valid - n_train

    # input, target, i_hidden, mask
    dataset = TensorDataset(torch.from_numpy(input).to(DEVICE,dtype=torch.float32),
                            torch.from_numpy(target).to(DEVICE,dtype=torch.float32),
                            torch.zeros(nconds,n_neurons).to(DEVICE,dtype=torch.float32),
                            mask)
    return random_split(dataset, [n_train, n_valid, n_test],
                        generator=torch.Generator().manual_seed(0))
    


    # train_dataset, valid_dataset, test_dataset
    # mask_t = mask_train[0,:,0]