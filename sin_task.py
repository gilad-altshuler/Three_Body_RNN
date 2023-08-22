import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_sin_data(n_batch,T):
    # random static vector x ~ U(0,1) #
    input = np.ones([n_batch, T],dtype="float32")*np.random.uniform(0,0.6,n_batch)[:,None]

    ####### defining the training data #######

    # 501 time steps - T=500, dt=0.01 #
    time_steps = np.linspace(0,5,T)

    # defining the frequency #
    omega = (input+0.01)
    # defining the sin wave #
    y = np.sin(2*np.pi*omega*time_steps)
    ######## convert data into Tensors ########
    input.resize((n_batch,T,1)) # batch_size,time_steps,input_size
    x_tensor = torch.tensor(input).float().to(DEVICE)

    y.resize((n_batch,T,1)) # time_steps,output_size
    y_tensor = torch.Tensor(y).to(DEVICE)

    return x_tensor , y_tensor

def plot_sin(T,Targets,prediction,loss=None,idx=0):
    time_steps = np.linspace(0,5,T)
    if(loss != None):
      print('Loss: ', loss.item())
    plt.figure(figsize=(14,5))
    plt.plot(time_steps, Targets[idx,:].detach(), 'g') # sin
    plt.plot(time_steps, prediction[idx].detach().numpy(), 'b') # predictions
    plt.show()
    plt.close()