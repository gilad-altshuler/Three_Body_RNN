import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

colors = np.array([[0.1254902, 0.29019608, 0.52941176],
                   [0.80784314, 0.36078431, 0.],
                   [0.30588235, 0.60392157, 0.02352941],
                   [0.64313725, 0., 0.]])
figsize = (8, 3)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_rand_neurons(model, n, hidden_dim, T, inputs):
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
    time_steps = np.linspace(0, T, T)
    N = hidden_dim

    idxs = np.random.choice(N, n)

    fig = plt.figure(figsize=(30, 10))
    ax = fig.subplots(1, 1)

    for i, idx in enumerate(idxs):
        ax.plot(time_steps, R[0, :, idx].cpu().detach().numpy(), c=colors[i], label="Index %d" % idx)

    ax.legend(loc=4)

    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(f'{n} random neurons activity')

    plt.show(fig)
    plt.close(fig)


def show_neurons(model, inputs, T, figsize=(30, 5)):
    """
    Show all neurons reflected from the model dynamics
    :param model: model to show
    :param inputs: inputs to show the hidden neurons in time
    :param T: timesteps
    :param figsize: optional to define the figure size
    :return: None
    """
    with torch.no_grad():
        _, _, R = model(inputs, None)
    # time_steps = np.linspace(0,T,T)
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)

    im = ax.imshow(R[0].permute(1, 0).cpu().detach().numpy(), aspect=1)
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


def show_pca(model, inputs, idx, steps=1):
    """
    :param model: TBRNN model to show 3D PCA of the first 3 trajectories PCs
    :param inputs: inputs to derive the trajectories from
    :param idx: index in input
    :param steps: steps in the PCs
    :param dyn: True if it is dynamical modelling, False otherwise
    :return: None
    """
    R = model(inputs, None)[2][idx].cpu()
    pca = PCA(n_components=3)
    pca.fit(R.detach().numpy())

    variance = pca.explained_variance_ratio_.cumsum()

    V = pca.transform(R.detach().numpy())

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot every steps
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

    return pca


def show_pca_2d(model, inputs, idx, steps=1, dyn=False):
    """
    :param model: TBRNN model to show 2D PCA of the first 2 trajectories PCs
    :param inputs: inputs to derive the trajectories from
    :param idx: index in input
    :param steps: steps in the PCs
    :param dyn: True if it is dynamical modelling, False otherwise
    :return: None
    """
    R = model(inputs, None)[2][idx].cpu()
    pca = PCA(n_components=2)
    pca.fit(R.detach().numpy())

    variance = pca.explained_variance_ratio_.cumsum()

    V = pca.transform(R.detach().numpy())

    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots(1, 1)
    ax.plot(V[::steps, 0], V[::steps, 1], '-o', c=colors[0])
    ax.plot(V[0, 0], V[0, 1], 'o', ms=7, c=colors[1], alpha=1, label="Init")
    ax.plot(V[-1, 0], V[-1, 1], '*', ms=10, c=colors[3], alpha=1, label="Final")
    ax.legend(loc=4)
    ax.axhline(0, c='0.5', zorder=-1)
    ax.set_ylabel("PC2")
    ax.set_xlabel("PC1")
    ax.set_title("2 components PCA: {var: .3g} explained variance".format(var=variance[1]), fontsize=10)
    plt.show(fig)
    plt.close(fig)

    return pca


def calc_jacobian(model, fixed_point, const_signal_tensor, model_type, model_mode):
    """
    :param model: dynamical model to perform forward pass on.
    :param fixed_point: some known fixed point with size of (hidden_dim)
    :param const_signal_tensor: size of (batch,T,input_size)
    :param mode: 'tbrnn' or 'rnn'
    :return: Jacobian at the given fixed point
    """
    from torch.autograd import Variable

    fixed_point = Variable(fixed_point.unsqueeze(dim=1)).to(DEVICE)
    fixed_point.requires_grad = True
    new_model = model.clone()
    new_model.M.requires_grad = False
    new_model.N.requires_grad = False
    if model_type == 'tbrnn':
        new_model.L.requires_grad = False

    activated = new_model(const_signal_tensor[:, [0], :], fixed_point[:, 0])[2][0].T
    if model_mode == 'cont':
        activated = (activated - fixed_point) / new_model.tau

    n_hid = new_model.N.shape[0]
    jacobian = torch.zeros(n_hid, n_hid)
    for i in range(n_hid):
        output = torch.zeros(n_hid, 1).to(DEVICE)
        output[i] = 1.
        jacobian[:, i:i + 1] = torch.autograd.grad(activated, fixed_point, grad_outputs=output, retain_graph=True)[0]

    jacobian = jacobian.numpy().T

    return jacobian


def eig_decomposition(M):
    fig = plt.figure(figsize=(7, 7))
    if len(M) <= 2:
        height = 1
        width = 2
    elif len(M) <= 4:
        height = width = 2
    elif len(M) <= 9:
        height = width = 3
    axes = fig.subplots(height,width)
    for i,m in enumerate(M):
        w, v = np.linalg.eig(m)
        w_real = list()
        w_im = list()
        for eig in w:
            w_real.append(round(eig.real,5))
            w_im.append(round(eig.imag,5))
        ax = axes.flatten()[i]
        ax.scatter(w_real, w_im)
        ax.set_xlabel(r'$Re(\lambda)$')
        ax.set_ylabel(r'$Im(\lambda)$')


def show_connectivity(model, mod="hr", dim=0):
    """
    :param model: TBRNN model to show its connectivity tensor W
    :param mod: "hr" if it is full rank, "lr" if low rank model
    :param dim: dimension of slicing
    :return: None
    """
    if mod == "hr":
        W = model.w_hh
    elif mod == "lr":
        W = model.lr_to_tensor()
    else:
        return
    hidden_dim = W.shape[0]
    fig = plt.figure(figsize=(20, 18))
    for i in range(hidden_dim):
        ax = fig.add_subplot(5, 6, i + 1)
        if dim == 0:
            ax.imshow(W[i].cpu().detach().numpy(), aspect=1)
        elif dim == 1:
            ax.imshow(W[:, i, :].cpu().detach().numpy(), aspect=1)
        elif dim == 2:
            ax.imshow(W[:, :, i].cpu().detach().numpy(), aspect=1)


def get_kappa_kappaI(model, Inputs, tau=0.2):
    """
    Use to compute the hidden whole network variables, kappa,kappaI, evolution in time

    :param model: low rank model to get the hidden kappas dynamics
    :param Inputs: batched inputs to the model (batch_size,T,input_size)
    :param tau: tau coefficient for K_I ODE, represent dt/tau
    :return: kappa (batch_size,T),kappaI (batch_size,T) evolution in time
    """
    from Models import Low_Rank_RNN_Dynamics, Low_Rank_Three_Way_RNN_Dynamics

    if isinstance(model, Low_Rank_Three_Way_RNN_Dynamics):
        lead = model.L
    elif isinstance(model, Low_Rank_RNN_Dynamics):
        lead = model.M
    else:
        return None, None

    _, _, traj = model(Inputs, None)
    kappa = ((traj @ lead)[:, :, 0] / (lead.norm() ** 2)).cpu().detach().numpy()

    Inputs = Inputs.cpu().detach().clone().numpy()
    kappaI = np.zeros_like(Inputs[:, :, 0])
    kappaI[:, 0] = Inputs[:, 0, 0]
    for t in range(1, len(Inputs[0, :, 0])):
        kappaI[:, t] = kappaI[:, t - 1] + tau * (-kappaI[:, t - 1] + Inputs[:, t, 0])

    return kappa, kappaI


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


def gram_schmidt(vecs, check_depend=True, epsilon=1e-5):
    vecs_orth = []
    vecs_orth.append(vecs[0] / np.linalg.norm(vecs[0]))
    for i in range(1, len(vecs)):
        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]
            if check_depend and (v < epsilon).all().item():  # then its linear dependent set
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
        subspace_orth = gram_schmidt(subspace_vecs, check_depend=False)
        M_proj = np.vstack(subspace_orth)
        projected_vecs.append(M_proj.T @ M_proj @ v)
    return projected_vecs


def angle_vectors(v, w):
    return np.arccos((v @ w) / (np.linalg.norm(v) * np.linalg.norm(w))) * 180 / np.pi

def project_L_orthogonal_to_I(L, I, b=None):
    # I: (N, d_u)
    # L: (N, R)
    if b is not None:
        I = torch.cat((I, b.unsqueeze(-1)), dim=1)
    I_gram = I.T @ I                 # (d_u, d_u)
    I_pinv = torch.linalg.pinv(I_gram) @ I.T  # (d_u, N)
    P_I = I @ I_pinv                 # (N, N)
    L_proj = L - P_I @ L             # (N, R)
    return L_proj

def lr_to_tensor_W(L, M, N):
    hidden, rank = L.shape
    W = np.zeros((hidden, hidden, hidden))
    for l, m, n in zip(L.T, M.T, N.T):
        W += (l.reshape(hidden, 1, 1) * np.outer(m, n).reshape(1, hidden, hidden))
    return W / (hidden ** 2)

def get_effective_tensor(model,normalize=True):

    if normalize:
        # normalize both tensors
        L,M,N = model.normalize_tensor()
    else:
        L,M,N = model.L,model.M,model.N

    L,M,N = L.cpu().detach().numpy(),M.cpu().detach().numpy(),N.cpu().detach().numpy()

    # compute the tensor W from L,M,N
    W = lr_to_tensor_W(L,M,N)

    if model.w_in.bias is not None:
        I = torch.cat((model.w_in.weight, model.w_in.bias.unsqueeze(-1)), dim=-1).cpu().detach().numpy()
    else:
        I = model.w_in.weight.cpu().detach().numpy()
    LI = np.concatenate((L,I),axis=1).T
    Mp = np.stack(project(M.T,LI)).T
    Np = np.stack(project(N.T,LI)).T

    W_eff = lr_to_tensor_W(L,Mp,Np)

    return W_eff, W

###### CKA helpers ##########

import gc

def center_gram(K):
    """Centering the Gram matrix."""
    n = K.size(0)
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    centered = H @ K @ H
    del H  # explicitly free
    return centered

def gram_linear(X):
    """Compute linear kernel Gram matrix."""
    return X @ X.T

def heavy_cka(X, Y):
    """Compute CKA (linear kernel) between X and Y."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    K = gram_linear(X)
    L = gram_linear(Y)
    
    K_centered = center_gram(K)
    L_centered = center_gram(L)

    hsic = (K_centered * L_centered).sum()
    norm_x = (K_centered * K_centered).sum()
    norm_y = (L_centered * L_centered).sum()

    del K, L, K_centered, L_centered
    gc.collect()
    torch.cuda.empty_cache()

    return hsic / (norm_x.sqrt() * norm_y.sqrt())

def CKA(X, Y):
    """
    Calculates CKA.
    """
    return (np.linalg.norm(X.T @ Y) ** 2) / np.linalg.norm(X.T @ X) / np.linalg.norm(Y.T @ Y)