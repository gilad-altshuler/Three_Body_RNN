"""
multifate task:

use N = 32
use symmetric,non-dimensional,expanded form
params:
- Kd = 1
- n = 1.5
- beta / alpha = 25 (we define them 90,3.6 respectively)

inputs take from one fp to another fp
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

def ode_solver(y0, input, dt=0.01, T=100, Kd=1, n=1.5, alpha=3.6, beta=90):
    from scipy.integrate import solve_ivp


    def ode(t, X):
        sum_X = X.sum()
        X2 = (2 * X**2) / (Kd + 4*sum_X + (Kd**2 + 8*Kd*sum_X)**0.5)
        # update protein states
        dX_dt = alpha - X + beta * (X2**n + input[math.floor(t/dt)]) / (1 + X2**n + input[math.floor(t/dt)])

        return dX_dt

    simT = int(T*dt)
    # Solve using ode15s (solve_ivp with 'BDF' method)
    solution = solve_ivp(
        fun=ode,
        t_span=(0, simT-0.0001),
        y0=y0,
        t_eval=np.arange(0,simT-0.0001,dt),
        method='Radau'
        # ,
        # rtol=1e-8,
        # atol=1e-8
    )
    return solution

import random

def generate_batched_init_states(B, P, N, alpha, beta):
    all_states = []

    for p in range(P):  # p = 0, 1, ..., P
        num_parts = p + 1
        for _ in range(B):
            # Generate random cut points for unequal partitions
            cuts = sorted([random.uniform(0, 1) for _ in range(num_parts - 1)])
            cuts = [0.0] + cuts + [1.0]
            partition_bounds = [(alpha + beta * cuts[i], alpha + beta * cuts[i+1]) for i in range(num_parts)]

            values = []
            for start_i, end_i in partition_bounds:
                # Choose random subinterval within this partition
                sub_len = end_i - start_i
                sub_start = random.uniform(start_i, end_i - 0.1 * sub_len)
                sub_end = random.uniform(sub_start + 0.01 * sub_len, end_i)

                # Number of values from this partition
                n_samples = N // num_parts
                samples = torch.FloatTensor(n_samples).uniform_(sub_start, sub_end)
                values.append(samples)

            x = torch.cat(values)

            # In case of rounding errors (e.g., N not divisible), pad with uniform samples
            if x.shape[0] < N:
                pad = torch.FloatTensor(N - x.shape[0]).uniform_(alpha, alpha + beta)
                x = torch.cat([x, pad])
            elif x.shape[0] > N:
                x = x[:N]

            # Shuffle and append
            x = x[torch.randperm(N)]
            all_states.append(x)

    return torch.stack(all_states)  # Shape: (P * B, N)

def generate_data(data_size, T, N ,dt = 0.1, Kd=1, n=1.5, alpha=3.6, beta=90, inducers=0, DEVICE="cpu"):
    """
    Generate synthetic data for the MultiFate task.
    :param data_size: Number of samples to generate
    :param T: Number of time steps
    :param N: Number of proteins
    :param dt: Time step size (default 0.1)
    :param Kd: Dissociation constant (default 1)
    :param n: Hill coefficient (default 1.5)
    :param alpha: Activation rate (default 0.4)
    :param beta: Degradation rate (default 10)
    :param inducers: Number of TFs input inducers to apply at each time step (default 0)
    :return: Tuple of input and target tensors
    """
    from concurrent.futures import ThreadPoolExecutor

    params = (dt, T, Kd, n, alpha, beta)
    random_data_size = 4000
    P = 5
    B = random_data_size // P
    init_state = generate_batched_init_states(B, P, N, alpha, beta)


    p = 0.0125               # probability to choose timstep

    # Initialize input array
    input = np.zeros((random_data_size, T, N), dtype=float)

    # Loop over random_data_size
    for b in range(random_data_size):
        # Sample which timesteps t to select (True with prob p)
        selected_t = np.zeros(T).astype(bool)
        t = 0
        while t<T:
            if np.random.rand() < p:
                selected_t[t] = True
                t += 15
            else:
                t += 1

        for t in np.where(selected_t)[0]:
            # Randomly choose inducers indices across N without replacement
            chosen_n = np.random.choice(N, size=inducers, replace=False)
            input[b, t:t+7, chosen_n] = 100.0

    # input in time 0 is always 0
    input[:,0,:] = 0.0

    # exec ode_solver in parallel
    with ThreadPoolExecutor() as executor:
        solutions = list(executor.map(lambda y,input: ode_solver(y,input,*params), init_state, input))
    
    target = torch.from_numpy(np.array([sol.y for sol in solutions])/(alpha+beta)).transpose(2,1).float().to(DEVICE)
    input = (torch.from_numpy(input)/(alpha+beta)).float().to(DEVICE)

    fps = (target[:,-1,:]>=0.2).sum(dim=-1).unique()
    print(fps)
    chosen = []

    for fp in fps:
      a = torch.nonzero((target[:,-1,:]>=0.2).sum(dim=-1) == fp).view(-1)
      print(f"TFs ON: {fp}, count: {len(a)}")
      chosen.append(a[torch.randperm(len(a))])

    chosen = torch.cat(chosen)
    chosen = chosen[torch.randperm(len(chosen))]

    input, x = input[chosen], target[chosen]

    #choose only fixed points with N/2 high expression proteins
    chosen = []
    x_half = torch.nonzero((x[:,-1,:]>=0.2).sum(dim=-1) == N//2).view(-1)

    chosen.append(x_half[torch.randperm(len(x_half))])
    chosen = torch.cat(chosen)
    chosen = chosen[torch.randperm(len(chosen))]


    return input[chosen[:data_size]], x[chosen[:data_size]]

def plot(input,target,prediction=None,idx=0):
    """
    Plot the target and prediction for the sine wave task.
    :param target: Target tensor of shape (batch_size, time_steps, output_size)
    :param prediction: Prediction tensor of shape (batch_size, time_steps, output_size)
    :param idx: Index of the sample to plot (default is 0)
    :return: None
    """

    fig = plt.figure(figsize=(14,5))
    if prediction is not None:
        ax = fig.subplots(1,2)
        ax[0,0].plot(target[idx].detach().cpu().numpy(), label='Target')
        ax[0,1].plot(prediction[idx].detach().cpu().numpy(), label='Prediction')
    else:
        ax = fig.subplots(1,1)
        ax.plot(target[idx].detach().cpu().numpy(), label='Target')
    fig.legend()
    fig.set_xlabel("Time")
    fig.set_ylabel("Protein concentration")
    fig.set_title("MultiFate Task")
    plt.show()
    plt.close()


def evaluate(model, input, target, hidden=None, r2_mode='per_neuron', r2_all=False):
    """
    Evaluate the model on the K-Bit Flipflop task.
    :param model: The model to evaluate
    :param input: Input tensor of shape (batch_size, time_steps, K)
    :param target: Target tensor of shape (batch_size, time_steps, K)
    :param teacher_traj: (Optional) tensor of teacher trajectories for evaluation. Default is None.
    :param rates: (Optional) If True, apply output nonlinearity to the trajectory. Default is False.
    :param r2_mode: Mode for R2 score calculation ('per_batch', 'per_time_step', or 'per_neuron'). Default is 'per_batch'.
    :param r2_all: If True, return R2 scores for all modes, otherwise return mean R2 score. Default is False.
    :param reduction: Reduction method for the loss ('mean', 'sum' or 'none'). Default is 'mean'.
    :return: Tuple of (accuracy, mse error, R2 score, and teacher-student trajectory mse error)
    """
    from sklearn.metrics import r2_score
    from methods.utils import cka_linear_streaming
    B,T,N = target.shape
    model.eval()
    with torch.no_grad():
        trajectory = model(input, hidden)[0]
        if r2_mode == 'per_batch':
            r2 = [r2_score(target[i].detach().cpu().numpy(), trajectory[i].detach().cpu().numpy()) for i in range(B)]
        elif r2_mode == 'per_time_step':
            r2 = [r2_score(target[:,i].detach().cpu().numpy(), trajectory[:,i].detach().cpu().numpy()) for i in range(T)]
        elif r2_mode == 'per_neuron':
            r2 = [r2_score(target[:,:,i].detach().cpu().numpy(), trajectory[:,:,i].detach().cpu().numpy()) for i in range(N)]
        else:
            raise ValueError("Invalid r2_mode. Choose 'per_batch', 'per_time_step' or 'per_neuron'.")
        if not r2_all:
            r2 = np.array(r2).mean().item()

        multifate = target.reshape(-1,N).detach().clone()
        trajectory = trajectory.reshape(-1,N).detach().clone()

        cka = cka_linear_streaming(multifate, trajectory).item()
        torch.cuda.empty_cache()
        
        return r2, cka