import torch
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
    """
    Same as show_neurons, but show a bunch of random neurons reflected from the model dynamics
    :param model: model to show
    :param n: number of random neurons
    :param inputs: inputs to show the hidden neurons in time
    :param T: timesteps
    :param inputs: inputs to show the hidden neurons in time
    :return: None
    """
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
    """
    Show all neurons reflected from the model dynamics
    :param model: model to show
    :param inputs: inputs to show the hidden neurons in time
    :param T: timesteps
    :param figsize: optional to define the figure size
    :return: None
    """
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

def show_pca(model,inputs,idx,steps = 1,dyn=False):
    """
    :param model: TBRNN model to show 3D PCA of the first 3 trajectories PCs
    :param inputs: inputs to derive the trajectories from
    :param idx: index in input
    :param steps: steps in the PCs
    :param dyn: True if it is dynamical modelling, False otherwise
    :return: None
    """
    R = model(inputs, None)[2][idx].cpu().transpose(0, 1)
    if dyn:
        R = model.nonlinearity(R)

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

def show_pca_2d(model,inputs,idx,steps = 1,dyn=False):
    """
    :param model: TBRNN model to show 2D PCA of the first 2 trajectories PCs
    :param inputs: inputs to derive the trajectories from
    :param idx: index in input
    :param steps: steps in the PCs
    :param dyn: True if it is dynamical modelling, False otherwise
    :return: None
    """
    R = model(inputs, None)[2][idx].cpu().transpose(0, 1)
    if dyn:
        R = model.nonlinearity(R)
    pca = PCA(n_components=30)
    pca.fit(R.detach().numpy().T)

    variance = pca.explained_variance_ratio_.cumsum()

    V = pca.transform(R.detach().numpy().T)

    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots(1,1)
    ax.plot(V[::steps, 0], V[::steps, 1], c=colors[0])
    ax.plot(V[0, 0], V[0, 1], 'o', ms=7, c=colors[1], alpha=1, label="Init")
    ax.plot(V[-1, 0], V[-1, 1], '*', ms=10, c=colors[3], alpha=1, label="Final")
    ax.legend(loc=4)
    ax.axhline(0, c='0.5', zorder=-1)
    ax.set_ylabel("PC2")
    ax.set_xlabel("PC1")
    ax.set_title("2 components PCA: {var: .3g} explained variance".format(var = variance[1]),fontsize=10)
    plt.show(fig)
    plt.close(fig)

def show_connectivity(model,mod = "hr",dim = 0):
    """
    :param model: TBRNN model to show its connectivity tensor W
    :param mod: "hr" if it is full rank, "lr" if low rank model
    :param dim: dimension of slicing
    :return: None
    """
    if mod == "hr":
        W = model.three_way_tensor
    elif mod == "lr":
        W = model.lr_to_tensor()
    hidden_dim = W.shape[0]
    fig = plt.figure(figsize=(20, 18))
    for i in range(hidden_dim):
        ax = fig.add_subplot(5, 6, i + 1)
        if dim == 0:
            ax.imshow(W[i].cpu().detach().numpy(), aspect=1)
        elif dim ==1:
            ax.imshow(W[:,i,:].cpu().detach().numpy(), aspect=1)
        elif dim == 2:
            ax.imshow(W[:, :, i].cpu().detach().numpy(), aspect=1)

def get_kappa_kappaI(model,Inputs,tau = 0.2):
    """
    Use to compute the hidden whole network variables, kappa,kappaI, evolution in time

    :param model: low rank model to get the hidden kappas dynamics
    :param Inputs: batched inputs to the model (batch_size,T,input_size)
    :param tau: tau coefficient for K_I ODE, represent dt/tau
    :return: kappa (batch_size,T),kappaI (batch_size,T) evolution in time
    """
    from Models import Low_Rank_RNN_Dynamics, Low_Rank_Three_Way_RNN_Dynamics

    if isinstance(model,Low_Rank_Three_Way_RNN_Dynamics):
        lead = model.L
    elif isinstance(model,Low_Rank_RNN_Dynamics):
        lead = model.M
    else:
        return None,None

    _,_,traj = model(Inputs,None)
    kappa = ((traj @ lead)[:,:,0] / (lead.norm()**2)).cpu().detach().numpy()

    Inputs = Inputs.cpu().detach().clone().numpy()
    kappaI = np.zeros_like(Inputs[:,:,0])
    kappaI[:,0] = Inputs[:,0,0]
    for t in range(1,len(Inputs[0,:,0])):
        kappaI[:,t] = kappaI[:,t-1] + tau * (-kappaI[:,t-1] + Inputs[:,t,0])

    return kappa,kappaI

####### Linear algebra helpers ##########

def gram_schmidt_pt(mat):
    """
    Performs INPLACE Gram-Schmidt
    :param mat:
    :return:
    """
    mat[0] = mat[0] / torch.norm(mat[0])
    for i in range(1, mat.shape[0]):
        mat[i] = mat[i] - (mat[:i].t() @ mat[:i] @ mat[i])
        mat[i] = mat[i] / torch.norm(mat[i])


def gram_schmidt(vecs,check_depend = True,epsilon=1e-5):
    vecs_orth = []
    vecs_orth.append(vecs[0] / np.linalg.norm(vecs[0]))
    for i in range(1, len(vecs)):
        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]
            if check_depend and (v < epsilon).all().item(): # then its linear dependent set
                return []
        v = v / np.linalg.norm(v)
        vecs_orth.append(v)
    return vecs_orth


def gram_factorization(G):
    """
    The rows of the returned matrix are the basis vectors whose Gramian matrix is G
    :param G: ndarray representing a symmetric semidefinite positive matrix
    :return: ndarray
    """
    w, v = np.linalg.eigh(G)
    x = v * np.sqrt(w)
    return x


def corrvecs(v, w):
    return v @ w / (np.linalg.norm(v) * np.linalg.norm(w))


def project(vecs, subspace_vecs):
    """
    :param vecs: vectors to be projected
    :param subspace_vecs: the projecting space vectors
    :return: projected_vecs: corresponding projected vectors list
    """
    projected_vecs = []
    for v in vecs:
        subspace_orth = gram_schmidt(subspace_vecs,check_depend = False)
        M_proj = np.vstack(subspace_orth)
        projected_vecs.append(M_proj.T @ M_proj @ v)
    return projected_vecs


def angle_vectors(v, w):
    return np.arccos((v @ w) / (np.linalg.norm(v) * np.linalg.norm(w))) * 180 / np.pi

def lr_to_tensor_W(L,M,N):
    hidden,rank = L.shape
    W = np.zeros((hidden, hidden, hidden))
    for l, m, n in zip(L.T, M.T, N.T):
        W += (l.reshape(hidden, 1, 1) * np.outer(m, n).reshape(1,hidden,hidden))
    return W / (hidden ** 2)


