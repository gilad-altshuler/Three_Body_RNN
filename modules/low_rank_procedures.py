from tensorly.decomposition import parafac,matrix_product_state
import tensorly as tl
import slicetca
import torch
from torch import nn
import sin_task
import PIN_task
from Models import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def TCA_procedure(model,Inputs,Targets,start_rank=1,end_rank=10,W0=None):
    """
    Apply low rank truncation according to Tensor Component Analysis (TCA) method.
    :param model: Full rank TBRNN to truncate its connectivity.
    :param Inputs: Inputs to evaluate the truncated model.
    :param Targets: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param W0: (optional) should be pass if try truncate delta W instead of W tensor.
    :return: accuracy array for each rank from start_rank to end_rank.
    """
    accs = []
    newModel = model.clone()
    for rank in range(start_rank,end_rank+1):
      # calculate TCA truncated accuracy
      twt = model.three_way_tensor.cpu().detach().clone().numpy()
      if W0 is not None:
          twt = twt - W0.cpu().detach().clone().numpy()
      factors = parafac(twt, rank=rank)
      twt = torch.tensor(tl.cp_to_tensor(factors),device=DEVICE)
      if W0 is not None:
          twt = twt + W0.detach().clone().to(DEVICE)
      newModel.three_way_tensor = torch.nn.Parameter(twt)
      accs.append(newModel.evaluate(Inputs,Targets).cpu().data)
    return accs


def TT_procedure(model,Inputs,Targets,start_rank=1,end_rank=10,W0=None):
    """
    Apply low rank truncation according to Tensor Train (TT) method.
    :param model: Full rank TBRNN to truncate its connectivity.
    :param Inputs: Inputs to evaluate the truncated model.
    :param Targets: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param W0: (optional) should be pass if try truncate delta W instead of W tensor.
    :return: accuracy array for each rank from start_rank to end_rank.
    """
    accs = []
    newModel = model.clone()
    for rank in range(start_rank,end_rank+1):
        twt = model.three_way_tensor.cpu().detach().clone().numpy()
        if W0 is not None:
            twt = twt - W0.cpu().detach().clone().numpy()
        tensor = tl.tensor(twt)
        factors = matrix_product_state(tensor, rank=[1,rank,30,1])
        twt = torch.tensor(tl.tt_to_tensor(factors),device=DEVICE)
        if W0 is not None:
            twt = twt + W0.detach().clone().to(DEVICE)
        newModel.three_way_tensor = torch.nn.Parameter(twt)
        accs.append(newModel.evaluate(Inputs,Targets).cpu().data)
    return accs


def sliceTCA_procedure(model,Inputs,Targets,start_rank=1,end_rank=10,W0=None):
    """
    Apply low rank truncation according to sliceTCA method (see paper - Dimensionality reduction beyond neural
    subspaces with slice tensor component analysis.
    Nature Neuroscience https://www.nature.com/articles/s41593-024-01626-2).
    :param model: Full rank TBRNN to truncate its connectivity.
    :param Inputs: Inputs to evaluate the truncated model.
    :param Targets: Targets to evaluate the truncated model.
    :param start_rank: rank to start the truncation from.
    :param end_rank: rank to stop the truncation method.
    :param W0: (optional) should be pass if try truncate delta W instead of W tensor.
    :return: accuracy array for each rank from start_rank to end_rank.
    """
    accs = []
    newModel = model.clone()

    twt = newModel.three_way_tensor.detach().clone()
    if W0 is not None:
        twt = twt - W0.detach().clone()

    for rank in range(start_rank, end_rank+1):
        _, mod = slicetca.decompose(twt / twt.std(), number_components=(rank, rank, rank))
        # For a not positive decomposition, we apply uniqueness constraints
        mod = slicetca.invariance(mod)
        reconstruction_full = mod.construct().detach() * twt.std()
        if W0 is not None:
            reconstruction_full = (reconstruction_full + W0.detach().clone()).to(DEVICE)
        newModel.three_way_tensor = torch.nn.Parameter(reconstruction_full)
        accs.append(newModel.evaluate(Inputs, Targets))
        torch.cuda.empty_cache()
    return accs

import K_Bit_Flipflop_task

def LINT_procedure(model,lr_class,Inputs,Targets,start_rank=1,end_rank=10,K=3,T=100,hidden_dim=30
                   ,n_steps=10000,batch_size = 128,train_size = 128,lr = 1e-03,to_save="",noise_std = 5e-2, tau=0.2):
    """
    :param model: Full rank model to perform the low rank approximation on.
    :param lr_class: Low rank Inference (LINT) approximation method.
    :param Inputs: Task input tensor.
    :param Targets: Task target tensor.
    :param start_rank: Initial connectivity rank to infer.
    :param end_rank: End connectivity rank to infer.
    :param K: Number of input channels.
    :param T: Timesteps.
    :param hidden_dim: Hidden dimension.
    :param n_steps: Number of train steps.
    :param batch_size: Train batch size.
    :param train_size:
    :param lr: learning rate.
    :param to_save: Path to save checkpoints
    :param noise_std: Noise STD to add for the model stochasticity.
    :param tau: Time scale tau.
    :return: Array of LINT accuracies.
    """
    # decide on hyperparameters
    input_size = K
    output_size = K

    # MSE loss and Adam optimizer with a learning rate
    criterion = nn.MSELoss().to(DEVICE)

    Train_Inputs, Train_Targets = K_Bit_Flipflop_task.generate_mem_data(train_size, T, K)
    accs = []
    for rank in range(start_rank, end_rank + 1):
        # instantiate an low rank RNN
        low_rank_rnn = lr_class(input_size, output_size, hidden_dim, rank, noise_std=noise_std, tau=tau).to(DEVICE)
        optimizer = torch.optim.Adam(low_rank_rnn.parameters(), lr=lr)
        _ = low_rank_rnn.lr_train(Train_Inputs, Train_Targets, model, n_steps, optimizer, criterion,
                                  batch_size, T)
        accs.append(low_rank_rnn.evaluate(Inputs, Targets))
        if to_save:
            torch.save(low_rank_rnn.state_dict(), to_save)
    return accs
