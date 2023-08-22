import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


colors = np.array([[0.1254902 , 0.29019608, 0.52941176],
       [0.80784314, 0.36078431, 0.        ],
       [0.30588235, 0.60392157, 0.02352941],
       [0.64313725, 0.        , 0.        ]])
figsize = (8, 3)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_rand_neurons(model,n,hidden_dim,T,inputs):
  with torch.no_grad():
    R = model(inputs, None)[2]
  time_steps = np.linspace(0,T,T)
  N = hidden_dim

  idxs = np.random.choice(N, n)

  fig = plt.figure(figsize=(30,10))
  ax = fig.subplots(1, 1)

  for i, idx in enumerate(idxs):
      ax.plot(time_steps, R[0,:,idx].cpu().detach().numpy(), c = colors[i], label="Index %d" % idx)


  ax.legend(loc=4)

  ax.set_xlabel("Time")
  ax.set_ylabel("Activity")
  ax.set_title(f'{n} random neurons activity')

  plt.show(fig)
  plt.close(fig)


def show_neurons(model,inputs,T, figsize=(30,5)):
  with torch.no_grad():
    _, _ , R = model(inputs, None)
  #time_steps = np.linspace(0,T,T)
  fig = plt.figure(figsize = figsize)
  ax = fig.subplots(1, 1)

  im = ax.imshow(R[0].permute(1,0).cpu().detach().numpy(), aspect=1)
  plt.colorbar(im, label="Activity")

  # Label x axis
  # So that the labels show the actual time.
  # nt = len(time_steps)
  # xticks = np.array([0, nt//2, nt-1])
  # ax.set_xticks(xticks)
  # ax.set_xticklabels(["%.1f" % tt for tt in time_steps[xticks]])

  ax.set_title("Neurons after train")
  ax.set_xlabel("Time")
  ax.set_ylabel("Neuron")
  plt.show(fig)
  plt.close(fig)

def show_pca(model,inputs,idx,steps = 1):

    R = model(inputs, None)[2][idx].cpu().transpose(0, 1)

    pca = PCA(n_components=30)
    pca.fit(R.detach().numpy().T)

    variance = pca.explained_variance_ratio_.cumsum()

    V = pca.transform(R.detach().numpy().T)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot every int_t = 10 steps
    ax.plot(V[::steps, 0], V[::steps, 1], V[::steps, 2], '-o', color=colors[0])

    plt.rcParams['axes.labelpad'] = 0.1
    ax.dist = 12

    ax.grid('off')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.set_title("3 components PCA: {var: .3g} explained variance".format(var=variance[2]), fontsize=10)

    plt.show()

def show_pca_2d(model,inputs,idx,steps = 1):
    R = model(inputs, None)[2][idx].cpu().transpose(0, 1)

    pca = PCA(n_components=30)
    pca.fit(R.detach().numpy().T)

    variance = pca.explained_variance_ratio_.cumsum()

    V = pca.transform(R.detach().numpy().T)

    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots(1,1)
    ax.plot(V[:steps, 0], V[::steps, 1], c=colors[0])
    ax.plot(V[0, 0], V[0, 1], 'o', ms=7, c=colors[1], alpha=1, label="Init")
    ax.plot(V[-1, 0], V[-1, 1], '*', ms=10, c=colors[3], alpha=1, label="Final")
    ax.legend(loc=4)
    ax.axhline(0, c='0.5', zorder=-1)
    ax.set_ylabel("PC2")
    ax.set_xlabel("PC1")
    ax.set_title("2 components PCA: {var: .3g} explained variance".format(var = variance[1]),fontsize=10)
    plt.show(fig)
    plt.close(fig)

def show_connectivity(model, hidden_dim, mod = "hr",dim = 0):
    if mod == "hr":
        W = model.three_way_tensor
    elif mod == "lr":
        W = model.W.view(30, 1, 1) * (model.U @ model.V.T).view(1, 30, 30)
    else:
        W = None
    fig = plt.figure(figsize=(20, 18))
    for i in range(hidden_dim):
        ax = fig.add_subplot(5, 6, i + 1)
        if dim == 0:
            ax.imshow(W[i].cpu().detach().numpy(), aspect=1)
        elif dim ==1:
            ax.imshow(W[:,i,:].cpu().detach().numpy(), aspect=1)
        elif dim == 2:
            ax.imshow(W[:, :, i].cpu().detach().numpy(), aspect=1)