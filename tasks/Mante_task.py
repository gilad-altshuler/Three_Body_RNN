"""
Code is adapted from:
Adrian Valente, May 2022.

Train low-rank networks on the Mante data.
"""

import pandas as pd
import torch
from torch.utils.data import random_split, TensorDataset
import numpy as np
from math import floor

## NEW SETUP: the fitted network is let free for the first 350 ms (while receiving contextual signals).
deltaT = 5
fixation_duration = 0
ctx_only_pre_duration = 350
stimulus_duration = 650   # counting first step
delay_duration = 80
decision_duration = 20


SCALE = 1
SCALE_CTX = 1
std_default = 1e-1
# decision targets
lo = -1
hi = 1


def setup():
    """
    Call this function whenever changing one of the global task variables (modifies other global variables)
    """
    global fixation_duration_discrete, stimulus_duration_discrete, ctx_only_pre_duration_discrete, \
        delay_duration_discrete, decision_duration_discrete, total_duration, stim_begin, stim_end, response_begin
    fixation_duration_discrete = floor(fixation_duration / deltaT)
    ctx_only_pre_duration_discrete = floor(ctx_only_pre_duration / deltaT)
    stimulus_duration_discrete = floor(stimulus_duration / deltaT)
    delay_duration_discrete = floor(delay_duration / deltaT)
    decision_duration_discrete = floor(decision_duration / deltaT)

    stim_begin = fixation_duration_discrete + ctx_only_pre_duration_discrete
    stim_end = stim_begin + stimulus_duration_discrete
    response_begin = stim_end + delay_duration_discrete
    total_duration = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete + \
                     ctx_only_pre_duration_discrete + decision_duration_discrete

def generate_mante_data_from_conditions(coherences_A, coherences_B, contexts, std=0):
    num_trials = coherences_A.shape[0]
    inputs_sensory = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    for i in range(num_trials):
        inputs[i, stim_begin:stim_end, 0] += coherences_A[i] * SCALE
        inputs[i, stim_begin:stim_end, 1] += coherences_B[i] * SCALE
        if contexts[i] == 1:
            inputs[i, fixation_duration_discrete:response_begin, 2] = 1. * SCALE_CTX
            targets[i, response_begin:] = hi if coherences_A[i] > 0 else lo
        elif contexts[i] == -1:
            inputs[i, fixation_duration_discrete:response_begin, 3] = 1. * SCALE_CTX
            targets[i, response_begin:] = hi if coherences_B[i] > 0 else lo
        mask[i, response_begin:, 0] = 1
    return inputs, targets, mask

def generate_data(DATA_DIR, monkey = 'A', DEVICE="cpu"):

    assert monkey in ['A', 'F'], "Monkey must be 'A' or 'F'"
    setup()

    modnum = 24  # Number of the low-rank fitted networks

    # Load preprocessed condition-averaged data
    conditions = pd.read_csv(DATA_DIR / f'conditions_monkey{monkey}.csv')
    X = np.load(DATA_DIR / f'X_cent_monkey{monkey}.npy')
    nconds, ntime, n_neurons = X.shape
    print(f"Training on monkey {monkey}, version {modnum}")
    print(X.shape)

    correct_trials = conditions.correct == 1
    input, _, _ = generate_mante_data_from_conditions(conditions[correct_trials]['stim_dir'].to_numpy(),
                                                            conditions[correct_trials]['stim_col'].to_numpy(),
                                                            conditions[correct_trials]['context'].to_numpy())
    target = np.concatenate([np.zeros((nconds, ctx_only_pre_duration_discrete, n_neurons)), X], axis=1)
    mask = torch.ones((nconds, ctx_only_pre_duration_discrete + ntime, n_neurons)).to(DEVICE,dtype=torch.bool)
    mask[:, :ctx_only_pre_duration_discrete, :] = False

    # Prepare training set, initial states for trajectories...
    n_train = int(0.8 * nconds)
    n_valid = int(0.1 * nconds)
    n_test = nconds - n_valid - n_train

    # input, target, i_hidden, mask
    dataset = TensorDataset(input.to(DEVICE),
                            torch.from_numpy(target).to(DEVICE,dtype=torch.float32),
                            torch.zeros(nconds,n_neurons).to(DEVICE,dtype=torch.float32),
                            mask)
    return random_split(dataset, [n_train, n_valid, n_test],
                        generator=torch.Generator().manual_seed(0))
    

def evaluate(model, dataset, r2_mode='per_batch', r2_all=False):
    """
    Evaluate the model on the K-Bit Flipflop task.
    :param model: The model to evaluate
    :param dataset: The dataset to evaluate on
    :param r2_mode: Mode for R2 score calculation ('per_batch', 'per_time_step', or 'per_neuron'). Default is 'per_batch'.
    :param r2_all: If True, return R2 scores for all modes, otherwise return mean R2 score. Default is False.
    :return: r2 score
    """
    from torch.utils.data import DataLoader
    from sklearn.metrics import r2_score
    for input, target, hidden, mask in DataLoader(dataset, len(dataset)): None
    B,T,N = target.shape
    model.eval()
    with torch.no_grad():
        trajectory = model(input, hidden)[0]
        trajectory = trajectory[mask].view(B,-1,N).detach().cpu().numpy()  # Apply mask to trajectory
        target = target[mask].view(B,-1,N).detach().cpu().numpy()
        if r2_mode == 'per_batch':
            return r2_score(target.ravel(), trajectory.ravel())
        elif r2_mode == 'per_time_step':
            r2 = [r2_score(target[:,i], trajectory[:,i]) for i in range(T)]
        elif r2_mode == 'per_neuron':
            target = target.transpose((2, 0, 1)).reshape((N, -1)).T
            trajectory = trajectory.transpose((2, 0, 1)).reshape((N, -1)).T
            r2 = [r2_score(target[:, i], trajectory[:, i]) for i in range(N)]
            return np.mean(r2)
        else:
            raise ValueError("Invalid r2_mode. Choose 'per_batch', 'per_time_step' or 'per_neuron'.")