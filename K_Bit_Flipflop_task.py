import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def draw_mem_charts(Inputs,Targets,Prediction=None,idx=0,K=2,T=100):
    fig = plt.figure(figsize=(15,10))
    axes = fig.subplots(K,1)
    inp = Inputs[idx].transpose(0,1)
    tar = Targets[idx].transpose(0,1)
    pred = Prediction[idx].transpose(0,1) if (Prediction!=None) else None
    time_steps = np.arange(T)
    for i in range(K):
      if K>1:
        ax = axes[i]
      else:
        ax = fig.add_subplot(1,1,1)
      ax.bar(time_steps, inp[i],label="spikes")
      ax.axhline()
      ax.plot(time_steps,tar[i],'--g',alpha=0.6,label="memory")
      ax.set_yticks([-1,0,1])
      ax.set_ylabel("Input {}".format(i+1))
      if(pred!=None):
          ax.plot(time_steps,pred[i].detach().numpy(),'-r',label="predicted")

    ax.set_xlabel("Time")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',fontsize=12)
    plt.show()
    plt.close()


def generate_mem_data(n_batch,T,K):
    categorical = torch.distributions.categorical.Categorical(torch.tensor([0.025, 0.95, 0.025]))
    t_in = categorical.sample((n_batch,T,K))-1
    t_in = torch.tensor(t_in,dtype=torch.float32)
    # T time steps
    X = t_in.transpose(1,2)
    Y = torch.zeros_like(X,dtype=torch.float32)

    for i_batch in range(n_batch):
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



