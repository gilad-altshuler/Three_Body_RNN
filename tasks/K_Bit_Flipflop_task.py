import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot(input,target,prediction=None,idx=0):
    """
    Draw memory charts for the K-Bit Flipflop task.
    :param input: Input tensor of shape (batch_size, time_steps, K)
    :param target: Target tensor of shape (batch_size, time_steps, K)
    :param prediction: Prediction tensor of shape (batch_size, time_steps, K) (optional)
    :param idx: Index of the sample to plot (default is 0)
    :return: None
    """
    K = input.shape[2]
    fig = plt.figure(figsize=(15,10))
    axes = fig.subplots(K,1)
    inp = input[idx].transpose(0,1)
    tar = target[idx].transpose(0,1)
    pred = prediction[idx].transpose(0,1) if (prediction!=None) else None
    time_steps = np.arange(input.shape[1])
    for i in range(K):
      if K>1:
        ax = axes[i]
      else:
        ax = fig.add_subplot(1,1,1)
      ax.bar(time_steps, inp[i],width=1.0, alpha=0.6,label="spikes")
      ax.axhline()
      ax.plot(time_steps,tar[i],'--g',linewidth=2, alpha=0.7,label="memory")
      ax.set_yticks([-1,0,1])

      ax.set_ylabel(f"Channel {i+1}", fontsize=14)
      ax.tick_params(labelsize=12)
      if(pred!=None):
          ax.plot(time_steps,pred[i].detach().numpy(),'-r',linewidth=2,label="predicted")

    ax.set_xlabel("Time",fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',fontsize=12,frameon=False)
    plt.show()
    plt.close()


def generate_data(data_size,T,K):
    """
    Generate synthetic data for the K-Bit Flipflop task.
    :param data_size: Number of samples to generate
    :param T: Number of time steps
    :param K: Number of input channels (bits)
    :return: Tuple of input and target tensors
    """
    categorical = torch.distributions.categorical.Categorical(torch.tensor([0.025, 0.95, 0.025]))
    t_in = categorical.sample((data_size,T,K)).float() - 1
    # T time steps
    X = t_in.transpose(1,2)
    Y = torch.zeros_like(X, dtype=torch.float)

    for i_batch in range(data_size):
        for i in range(K):
            # random first current
            last_one = X[i_batch,i,0] if X[i_batch,i,0] else np.random.randint(0,2)*2-1
            X[i_batch,i,0] = last_one
            for t in range(T):
                last_one = X[i_batch,i,t] if X[i_batch,i,t] else last_one
                Y[i_batch,i,t] = last_one

    X = X.transpose(1,2).to(DEVICE)
    Y = Y.transpose(1,2).to(DEVICE)
    return X , Y

def evaluate(model, input, target, teacher_traj=None, rates=False, r2_mode='per_batch', r2_all=False):
    """
    Evaluate the model on the K-Bit Flipflop task.
    :param model: The model to evaluate
    :param input: Input tensor of shape (batch_size, time_steps, K)
    :param target: Target tensor of shape (batch_size, time_steps, K)
    :param teacher_traj: (Optional) tensor of teacher trajectories for evaluation. Default is None.
    :param rates: (Optional) If True, apply output nonlinearity to the trajectory. Default is False.
    :param r2_mode: Mode for R2 score calculation ('per_batch', 'per_time_step', or 'per_neuron'). Default is 'per_batch'.
    :param r2_all: If True, return R2 scores for all modes, otherwise return mean R2 score. Default is False.
    :return: Tuple of (accuracy, mse error, R2 score, and teacher-student trajectory mse error)
    """
    from sklearn.metrics import r2_score
    
    model.eval()
    with torch.no_grad():
        prediction, _, trajectory = model(input, None)
        if rates:
            trajectory = model.output_nonlinearity(trajectory)

        acc = accuracy(prediction,target)
        error = torch.nn.functional.mse_loss(prediction, target).item()
        if teacher_traj is not None:
            if r2_mode == 'per_batch':
                r2 = np.array([r2_score(teacher_traj[i].detach().cpu().numpy(), trajectory[i].detach().cpu().numpy()) for i in range(len(teacher_traj))])
            elif r2_mode == 'per_time_step':
                r2 = np.array([r2_score(teacher_traj[:,i].detach().cpu().numpy(), trajectory[:,i].detach().cpu().numpy()) for i in range(teacher_traj.shape[1])])
            elif r2_mode == 'per_neuron':
                r2 = np.array([r2_score(teacher_traj[:,:,i].detach().cpu().numpy(), trajectory[:,:,i].detach().cpu().numpy()) for i in range(teacher_traj.shape[2])])
            else:
                raise ValueError("Invalid r2_mode. Choose 'per_batch', 'per_time_step' or 'per_neuron'.")
            if not r2_all:
                r2 = r2.mean().item()
            ts_error = torch.nn.functional.mse_loss(trajectory, teacher_traj).item()
            return acc,error,r2,ts_error
        else:
            return acc,error,None,None
        

def accuracy(prediction,target):
    sign_prediction = prediction.sign()
    acc = (sign_prediction == target).float().mean().item()
    return acc


