import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_General_PE_ODE(Proteins, Complexes, K1, Km1, K2, T, delta):
    N = len(K1)
    mask = np.ones((N,N,N))
    for j in range(N):
      mask[:,j,:] = 1 - np.eye(N)


    for t in range(T-1):
      proteins = Proteins[:,t]
      complexes = Complexes[:,:,t]
      Proteins[:,t+1] = proteins \
                      + ((Km1.reshape(N,1,N) @ complexes.reshape(N,N,1) + Km1.T.reshape(N,1,N) @ complexes.T.reshape(N,N,1)).flatten() \
                      - proteins * (K1 @ proteins + K1.T @ proteins) \
                      + (K2.transpose(2,0,1) * complexes).sum((1,2)) \
                      + ((mask * K2).sum(2).reshape(N,1,N) @ complexes.reshape(N,N,1)).flatten()) * delta

      Complexes[:,:,t+1] = complexes \
                        + ((proteins.reshape(N,1) @ proteins.reshape(1,N) * K1) \
                        - (complexes * (Km1 + K2.sum(2)))) * delta

def generate_PIN_data(n_batch,T,delta,K1,Km1,K2):
    # Autonomous mode #
    N = len(K1)
    B_Proteins = np.zeros((n_batch, N, T))
    ####### defining the training data #######
    for i in range(n_batch):
        Proteins = np.zeros((N, T))
        Complexes = np.zeros((N, N, T))
        Proteins[:, 0] = 10*np.random.rand(N)

        generate_General_PE_ODE(Proteins,Complexes,K1,Km1,K2,T,delta)
        B_Proteins[i,:,:] = Proteins

    B_Proteins = B_Proteins/10
    B_init_Proteins = np.zeros((n_batch,T,N))
    B_Proteins = B_Proteins.transpose((0,2,1)) # ibatch,time_steps,output_size
    B_init_Proteins[:, 0, :] = B_Proteins[:, 0, :]
    y_tensor = torch.tensor(B_Proteins).float().to(DEVICE)
    x_tensor = torch.tensor(B_init_Proteins).float().to(DEVICE)

    return x_tensor, y_tensor

def plot_proteins(Proteins,Protein_pred=[]):
  for i,protein in enumerate(Proteins):
    _= plt.plot(protein,label=f'Protein {i}')
  for i,protein in enumerate(Protein_pred):
    _= plt.plot(protein,linestyle='dashed',label=f'Predicted Protein {i}')

  _= plt.xlabel('Time (steps)')
  _= plt.ylabel('Concentration (nM)')
  _= plt.title('Proteins concentration in time')
  plt.legend()
  plt.show()