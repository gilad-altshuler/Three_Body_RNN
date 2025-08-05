import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_data(data_size=128,T=100,input_size=1, DEVICE="cpu"):

    eps = 0.04
    # random static vector x ~ U(0,1) #
    input = np.ones([data_size, T],dtype="float32")*np.random.uniform(eps,1.0,data_size)[:,None]

    ####### defining the training data #######

    time_steps = np.linspace(0,5*(T//100),T*(T//100))

    # defining the frequency #
    omega = (0.4*input+0.1)
    # defining the sin wave #
    y = np.sin(2*np.pi*omega*time_steps)
    ######## convert data into Tensors ########
    input.resize((data_size,T,input_size)) # batch_size,time_steps,input_size
    x_tensor = torch.tensor(input).float().to(DEVICE)

    y.resize((data_size,T,input_size)) # time_steps,output_size
    y_tensor = torch.Tensor(y).to(DEVICE)

    return x_tensor , y_tensor

def plot(input,target,prediction=None,idx=0):
    """
    Plot the target and prediction for the sine wave task.
    :param target: Target tensor of shape (batch_size, time_steps, output_size)
    :param prediction: Prediction tensor of shape (batch_size, time_steps, output_size)
    :param idx: Index of the sample to plot (default is 0)
    :return: None
    """
    time_steps = np.linspace(0,5,target.shape[1])
    plt.figure(figsize=(14,5))
    plt.plot(time_steps, target[idx,:].detach(), 'g') # sin
    if prediction is not None:
      plt.plot(time_steps, prediction[idx].detach().numpy(), 'b') # predictions
    plt.show()
    plt.close()

def evaluate(model, input, target, teacher_traj=None, rates=False, r2_mode='per_batch', r2_all=False,reduction='mean'):
    """
    Evaluate the model on the Sine prediction task.
    :param model: The model to evaluate
    :param input: Input tensor of shape (batch_size, time_steps, K)
    :param target: Target tensor of shape (batch_size, time_steps, K)
    :param teacher_traj: (Optional) tensor of teacher trajectories for evaluation. Default is None.
    :param rates: (Optional) If True, apply output nonlinearity to the model trajectory. Default is False.
    :param r2_mode: Mode for R2 score calculation ('per_batch', 'per_time_step', or 'per_neuron'). Default is 'per_batch'.
    :param r2_all: If True, return R2 scores for all modes, otherwise return mean R2 score. Default is False.
    :param reduction: Reduction method for the loss ('mean', 'sum' or 'none'). Default is 'mean'.
    :return: Tuple of (accuracy, mse error, R2 score, and teacher-student trajectory mse error) if teacher_traj is not None, otherwise (accuracy, mse error, None, None)
    """
    from sklearn.metrics import r2_score
    
    model.eval()
    with torch.no_grad():
        prediction, _, trajectory = model(input, None)
        if rates:
          trajectory = model.output_nonlinearity(trajectory)
          teacher_traj = model.output_nonlinearity(teacher_traj)
        acc = accuracy(prediction,target)
        error = torch.nn.functional.mse_loss(prediction, target,reduction=reduction)
        if reduction == 'none':
            error = error.mean(axis=(1,2)).detach().cpu().numpy()
        else:
            error = error.item()
        if teacher_traj is not None:
            if r2_mode == 'per_batch':
                r2 = [r2_score(teacher_traj[i].detach().cpu().numpy(), trajectory[i].detach().cpu().numpy()) for i in range(len(teacher_traj))]
            elif r2_mode == 'per_time_step':
                r2 = [r2_score(teacher_traj[:,i].detach().cpu().numpy(), trajectory[:,i].detach().cpu().numpy()) for i in range(teacher_traj.shape[1])]
            elif r2_mode == 'per_neuron':
                r2 = [r2_score(teacher_traj[:,:,i].detach().cpu().numpy(), trajectory[:,:,i].detach().cpu().numpy()) for i in range(teacher_traj.shape[2])]
            else:
                raise ValueError("Invalid r2_mode. Choose 'per_batch', 'per_time_step' or 'per_neuron'.")
            if not r2_all:
                r2 = np.array(r2).mean().item()

            ts_error = torch.nn.functional.mse_loss(trajectory, teacher_traj).item()
            return acc,error,r2,ts_error
        else:
            return acc,error,None,None
        
def accuracy(prediction,target):
    """
    Calculate the accuracy of the model's predictions.
    :param prediction: Prediction tensor of shape (batch_size, time_steps, output_size)
    :param target: Target tensor of shape (batch_size, time_steps, output_size)
    :return: General accuracy as a float
    """
    sign_prediction = prediction.sign()
    sign_target = target.sign()
    acc = (sign_prediction == sign_target).float().mean().item()
    return acc