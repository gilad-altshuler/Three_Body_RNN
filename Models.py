import importlib
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt

MODELS = ["rnn", "tbrnn", "hornn", "gru"]

def get_model_str(model_class):
    """
    Get a string representation of the model class based on its name.
    :param model_class (type): The class of the model.
    :return: A string representation of the model class.
    """
    model_name = model_class.__name__.lower()  # Convert class name to lowercase
    for keyword in sorted(MODELS, key=len, reverse=True):
        if keyword in model_name:
            return keyword
    return None  # Return None if none of the keywords are found

def train(model,input,target,epochs,optimizer,criterion,
          scheduler=None,mask_train=None,batch_size=128,hidden=None,
          clip_gradient=None,keep_best=True,plot=False):
    """ 
    Train the model on the provided input and target data.
    :param model: The model to be trained.
    :param input: Input tensor of shape (batch_size, time_steps, input_size).
    :param target: Target tensor of shape (batch_size, time_steps, output_size).
    :param epochs: Number of epochs to train the model.
    :param optimizer: Optimizer to use for training.
    :param criterion: Loss function to use for training.
    :param scheduler: Learning rate scheduler (optional).
    :param mask_train: Mask for training data (optional).
    :param batch_size: Batch size for training.
    :param hidden: Initial hidden state for the model of shape (batch_size, hidden_dim) (optional).
    :param clip_gradient: Gradient clipping value (optional).
    :param keep_best: Whether to keep the best model based on validation loss.
    :param plot: Whether to visualize the training progress.
    :return: List of losses recorded during training.
    """
    B,T,_ = input.shape
    N = model.hidden_dim
    DEVICE = next(model.parameters()).device
    if hidden is None:
        hidden = torch.zeros(B,N).to(DEVICE)
    dataset = TensorDataset(input,target,hidden)
    losses = []
    best_loss = torch.tensor(float('inf')).to(DEVICE)
    last_epoch_best_loss = best_loss
    eps = 1e-4

    if mask_train is not None:
        mask_t = mask_train[0,:,0]
    else:
        mask_t = torch.ones(T,device=DEVICE)

    for epoch in tqdm(range(epochs)):
        dataloader = DataLoader(dataset,batch_size,shuffle=True)
        for X, Y, b_hidden in dataloader:
            optimizer.zero_grad()
            prediction, _, _ = model.forward(X, b_hidden)
            ### calculate the loss
            loss = criterion(prediction.transpose(0,1)[mask_t==1].transpose(0,1), Y.transpose(0,1)[mask_t==1].transpose(0,1))
            ### perform backprop and update weights
            loss.backward()
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        torch.cuda.empty_cache()
        # display loss and predictions
        with torch.no_grad():
            prediction,_ ,_ = model.forward(input, hidden)
            loss = criterion(prediction.transpose(0,1)[mask_t==1].transpose(0,1), target.transpose(0,1)[mask_t==1].transpose(0,1))
            losses.append(loss.cpu().data.item())
            if best_loss > loss:
                model.best_model = copy.deepcopy(model.state_dict())
                best_loss = loss

            if loss < eps:
                print('Early stopping at epoch {} with loss {:.4f}'.format(epoch, loss.cpu().data.item()))
                break

        if epoch % (epochs/10) == 0:
            print(int(epoch / (epochs / 10)),
                    '/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
            
            if last_epoch_best_loss - best_loss < eps:
                print('No improvement in the last 10%, stopping training.')
                break
            else:
                last_epoch_best_loss = best_loss

            if plot:
                plot_func = getattr(importlib.import_module(f"tasks.{model.task}"), 'plot')
                plot_func(input.cpu(),target.cpu(),prediction.cpu(),idx=0)

    if keep_best:
        model.load_state_dict(model.best_model)

    with torch.no_grad():
        prediction, _ , _ = model.forward(input, hidden)
        loss = criterion(prediction.transpose(0,1)[mask_t==1].transpose(0,1), target.transpose(0,1)[mask_t==1].transpose(0,1)).cpu()
        print('Loss: ', loss.data.item())

    return losses


class TBRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity=torch.tanh, output_nonlinearity=torch.tanh, 
                 task="", mode="cont", form="rate", noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, 
                 w_out = None, w_in = None, hard_orth=False):
        super(TBRNN, self).__init__()

        self.hidden_dim=hidden_dim

        #self.w_hh = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,hidden_dim))))
        self.w_hh = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim,hidden_dim))
        nn.init.normal_(self.w_hh, std=.1 / (hidden_dim))

        if w_in is not None:
            self.w_in = w_in
        else:
            self.w_in = nn.Linear(input_size, hidden_dim, bias = Win_bias)
            #nn.init.xavier_uniform_(self.w_in.weight)  ###init
            nn.init.normal_(self.w_in.weight)
            if Win_bias:
                nn.init.zeros_(self.w_in.bias)

        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity
        
        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, input_size, bias = Wout_bias)

        self.task = task

        if mode not in ['cont', 'disc']:
            raise Exception("Error: Mode does not exists.")
        self.mode = mode

        if form not in ['rate', 'voltage']:
            raise Exception("Error: Form does not exists.")
        self.form = form

        self.input_size = input_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth


    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # hidden (batch_size, hidden_dim)
        # r_out (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t, u_t in enumerate(u.permute(1, 0, 2)):
            if self.form == 'voltage':
                r = x
            elif self.form == 'rate':
                r = self.nonlinearity(x)

            # calculating: input weight
            input_I = self.w_in(u_t)

            hidden_W = (r.view(b_size,1,1,hid) @ self.w_hh.view(1,hid,hid,hid) @ r.view(b_size,1,hid,1)).reshape((b_size,hid))

            if self.form == 'voltage':
                rec_x = self.nonlinearity(hidden_W + input_I)
            elif self.form == 'rate':
                rec_x = hidden_W + input_I

            # update hidden variable x accrding to the currect mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x

            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1, 0, 2)

        # get final output
        if self.form == 'voltage':
            output = self.w_out(trajectories)
        elif self.form == 'rate':
            output = self.w_out(self.output_nonlinearity(trajectories))

        return output, x, trajectories



    def clone(self):
        new_net = TBRNN(self.input_size, self.output_size, self.hidden_dim, self.nonlinearity,self.output_nonlinearity,
                         self.task, self.mode, self.form, self.noise_std, self.tau, self.Win_bias, self.Wout_bias).to(next(self.parameters()).device)
        new_net.w_in = copy.deepcopy(self.w_in)
        new_net.w_hh = nn.Parameter(self.w_hh.detach().clone())
        new_net.w_out = copy.deepcopy(self.w_out)
        return new_net

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity=torch.tanh, output_nonlinearity=torch.tanh, 
                 task="", mode="cont", form="rate", noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, 
                 w_out = None, w_in = None,
                 hard_orth=False):
        super(RNN, self).__init__()

        self.hidden_dim=hidden_dim

        #self.w_hh = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,hidden_dim))))
        self.w_hh = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim))
        nn.init.normal_(self.w_hh, std=.1 / (hidden_dim**0.5))
        
        if w_in is not None:
            self.w_in = w_in
        else:
            self.w_in = nn.Linear(input_size, hidden_dim, bias = Win_bias)
            #nn.init.xavier_uniform_(self.w_in.weight)  ###init
            nn.init.normal_(self.w_in.weight)
            if Win_bias:
                nn.init.zeros_(self.w_in.bias)

        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity
        
        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, input_size, bias = Wout_bias)

        self.task = task

        if mode not in ['cont', 'disc']:
            raise Exception("Error: Mode does not exists.")
        self.mode = mode

        if form not in ['rate', 'voltage']:
            raise Exception("Error: Form does not exists.")
        self.form = form

        self.input_size = input_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t, u_t in enumerate(u.permute(1, 0, 2)):
            if self.form == 'voltage':
                r = x
            elif self.form == 'rate':
                r = self.nonlinearity(x)

            # calculating: input weight
            input_I = self.w_in(u_t)

            hidden_W =  r @ self.w_hh.T #/ (self.hidden_dim if self.mode == 'cont' else 1.0)

            if self.form == 'voltage':
                rec_x = self.nonlinearity(hidden_W + input_I)
            elif self.form == 'rate':
                rec_x = hidden_W + input_I

            # update hidden variable x accrding to the currect mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x

            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1, 0, 2)

        # get final output
        if self.form == 'voltage':
            output = self.w_out(trajectories)
        elif self.form == 'rate':
            output = self.w_out(self.output_nonlinearity(trajectories))

        return output, x, trajectories

    def clone(self):
        new_net = RNN(self.input_size, self.output_size, self.hidden_dim, self.nonlinearity, self.output_nonlinearity,
                                 self.task, self.mode, self.form, self.noise_std, self.tau, self.Win_bias, self.Wout_bias).to(next(self.parameters()).device)
        new_net.w_in = copy.deepcopy(self.w_in)
        new_net.w_hh = nn.Parameter(self.w_hh.detach().clone())
        new_net.w_out = copy.deepcopy(self.w_out)
        return new_net


class Low_Rank_TBRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity=torch.tanh, 
                 output_nonlinearity=torch.tanh, task="", mode="cont", form="rate",
                 noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, w_out = None, w_in = None,
                 hard_orth=False):
        super(Low_Rank_TBRNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define a low rank RNN
        self.L = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))
        self.M = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))
        self.N = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))

        if w_in is not None:
            self.w_in = w_in
        else:
            self.w_in = nn.Linear(input_size, hidden_dim, bias = Win_bias)
            nn.init.xavier_uniform_(self.w_in.weight)

        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, output_size, bias = Wout_bias)
            nn.init.xavier_uniform_(self.w_out.weight)

        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.task = task

        if mode not in ['cont','disc']:
            raise Exception("Error: Mode does not exists.")
        self.mode = mode

        if form not in ['rate','voltage']:
            raise Exception("Error: Form does not exists.")
        self.form = form

        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t,u_t in enumerate(u.permute(1, 0, 2)):
            if self.form == 'voltage':
                r = x
            elif self.form == 'rate':
                r = self.nonlinearity(x)
            # calculating: input weight
            input_I = self.w_in(u_t)

            if self.hard_orth:
                L = utils.project_L_orthogonal_to_I(self.L,self.w_in.weight,self.w_in.bias)
            else:
                L = self.L
            # calculating: sum h_t^T @ (1/N^2 * L_r @ M_r @ N_r) @ ht
            hidden_W = ((r @ self.M) * (r @ self.N)) @ L.T / (self.hidden_dim**2 if self.mode == 'cont' else 1.0)

            if self.form == 'voltage':
                rec_x = self.nonlinearity(hidden_W + input_I)
            elif self.form == 'rate':
                rec_x = hidden_W + input_I

            # update hidden variable x accrding to the currect mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x
        
            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1,0,2)

        # get final output
        if self.form == 'voltage':
            output = self.w_out(trajectories)
        elif self.form == 'rate':
            output = self.w_out(self.output_nonlinearity(trajectories))

        return output, x, trajectories

    def lr_to_tensor(self):
        L = self.L.T.cpu().detach()
        M = self.M.T.cpu().detach()
        N = self.N.T.cpu().detach()
        W = torch.zeros(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        for l, m, n in zip(L, M, N):
            W += (l.view(self.hidden_dim, 1, 1) * torch.outer(m, n).unsqueeze(0))
        return W / (self.hidden_dim ** 2)
    
    def clone(self):
        new_net = Low_Rank_TBRNN(self.input_size, self.output_size, self.hidden_dim, self.rank, self.nonlinearity, self.output_nonlinearity,
                                 self.task, self.mode, self.form, self.noise_std, self.tau, self.Win_bias, self.Wout_bias).to(next(self.parameters()).device)
        new_net.L = nn.Parameter(self.L.detach().clone())
        new_net.M = nn.Parameter(self.M.detach().clone())
        new_net.N = nn.Parameter(self.N.detach().clone())
        new_net.w_in = copy.deepcopy(self.w_in)
        new_net.w_out = copy.deepcopy(self.w_out)
        return new_net

    def normalize_tensor_(self): #inplace function
        L,M,N = self.normalize_tensor()
        self.L, self.M, self.N = nn.Parameter(L), nn.Parameter(M), nn.Parameter(N)
        return L,M,N

    def normalize_tensor(self):
        # first, check that vecs are independent in each L,M,N matrices
        from utils import gram_schmidt
        if not gram_schmidt(self.L.T.cpu().detach().clone().numpy()):
            raise Exception(f"L vectors are linear dependent, cannot normalize")
        if not gram_schmidt(self.M.T.cpu().detach().clone().numpy()):
            raise Exception(f"M vectors are linear dependent, cannot normalize")
        if not gram_schmidt(self.N.T.cpu().detach().clone().numpy()):
            raise Exception(f"N vectors are linear dependent, cannot normalize")

        # Now make the tensor unique and canonical
        norml, normm, normn = self.L.norm(dim=0), self.M.norm(dim=0), self.N.norm(dim=0)
        coef = (norml * normm * normn) ** (1 / 3)
        _, indices = coef.sort()
        if normm>=normn:
            return (self.L * coef / norml)[:, indices], (self.M * coef / normm)[:, indices], (self.N * coef / normn)[:, indices]
        else: 
            return (self.L * coef / norml)[:, indices], (self.N * coef / normn)[:, indices], (self.M * coef / normm)[:, indices]
class Low_Rank_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity=torch.tanh, 
                 output_nonlinearity=torch.tanh, task="", mode="cont", form="rate",
                 noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, w_out = None, w_in = None,
                 hard_orth=False):
        super(Low_Rank_RNN, self).__init__()

        self.task = task
        self.hidden_dim=hidden_dim

        # define a low rank RNN
        self.M = nn.Parameter(nn.init.normal_(torch.empty(hidden_dim,rank)))
        self.N = nn.Parameter(nn.init.normal_(torch.empty(hidden_dim,rank)))

        if w_in is not None:
            self.w_in = w_in
        else:
            self.w_in = nn.Linear(input_size, hidden_dim, bias = Win_bias)
            nn.init.xavier_uniform_(self.w_in.weight)

        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, output_size, bias = Wout_bias)
            nn.init.xavier_uniform_(self.w_out.weight)


        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity

        if mode not in ['cont', 'disc']:
            raise Exception("Error: Mode does not exists.")
        self.mode = mode

        if form not in ['rate', 'voltage']:
            raise Exception("Error: Form does not exists.")
        self.form = form

        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t, u_t in enumerate(u.permute(1, 0, 2)):
            if self.form == 'voltage':
                r = x
            elif self.form == 'rate':
                r = self.nonlinearity(x)

            # calculating: input weight
            input_I = self.w_in(u_t)

            hidden_W = r @ self.N @ self.M.T / (self.hidden_dim if self.mode == 'cont' else 1.0)

            if self.form == 'voltage':
                rec_x = self.nonlinearity(hidden_W + input_I)
            elif self.form == 'rate':
                rec_x = hidden_W + input_I

            # update hidden variable x accrding to the currect mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x

            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1, 0, 2)

        # get final output
        if self.form == 'voltage':
            output = self.w_out(trajectories)
        elif self.form == 'rate':
            output = self.w_out(self.output_nonlinearity(trajectories))

        return output, x, trajectories

    def lr_to_tensor(self):
        M = self.M.T.cpu().detach()
        N = self.N.T.cpu().detach()
        W = torch.zeros(self.hidden_dim, self.hidden_dim)
        for m, n in zip(M, N):
            W += torch.outer(m, n)
        return W / self.hidden_dim

    def clone(self):
        new_net = Low_Rank_RNN(self.input_size, self.output_size, self.hidden_dim, self.rank, self.nonlinearity, self.output_nonlinearity,
                                 self.task, self.mode, self.form, self.noise_std, self.tau, self.Win_bias, self.Wout_bias).to(next(self.parameters()).device)
        new_net.M = nn.Parameter(self.M.detach().clone())
        new_net.N = nn.Parameter(self.N.detach().clone())
        new_net.w_in = copy.deepcopy(self.w_in)
        new_net.w_out = copy.deepcopy(self.w_out)
        return new_net

    def normalize_tensor(self):
        with torch.no_grad():
            structure = (self.M @ self.N.t()).cpu().detach().numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            return torch.from_numpy(m * np.sqrt(s)).to(next(self.parameters()).device),torch.from_numpy(n.transpose() * np.sqrt(s)).to(next(self.parameters()).device)

    def normalize_tensor_(self):
        m, n = self.normalize_tensor()
        self.M.set_(m)
        self.N.set_(n)
        return m,n


class Low_Rank_GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity=torch.tanh, output_nonlinearity=torch.tanh,
                 task="", mode="cont", form="rate", noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, w_out=None,
                 hard_orth=False):
        super(Low_Rank_GRU, self).__init__()

        self.hidden_dim = hidden_dim

        # define a low rank parameters for GRU
        self.M_hr = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))
        self.N_hr = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))

        self.M_hz = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))
        self.N_hz = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))

        self.M_hh = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))
        self.N_hh = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank)))

        self.W_ir = nn.Linear(input_size, hidden_dim, bias=Win_bias)
        self.W_iz = nn.Linear(input_size, hidden_dim, bias=Win_bias)
        self.W_ih = nn.Linear(input_size, hidden_dim, bias=Win_bias)

        nn.init.xavier_uniform_(self.W_ir.weight)  ###init
        nn.init.xavier_uniform_(self.W_iz.weight)  ###init
        nn.init.xavier_uniform_(self.W_ih.weight)  ###init

        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity


        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, output_size, bias = Wout_bias)
            nn.init.xavier_uniform_(self.w_out.weight)

        self.task = None

        if mode not in ['cont', 'disc']:
            raise Exception("Error: Mode does not exists.")
        
        self.task = task
        self.mode = mode
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t, u_t in enumerate(u.permute(1, 0, 2)):
            # calculating: input weight
            ir = self.W_ir(u_t)
            iz = self.W_iz(u_t)
            ih = self.W_ih(u_t)

            hr = (x @ self.N_hr) @ self.M_hr.T / (self.hidden_dim if self.mode == 'cont' else 1.0)
            hz = (x @ self.N_hz) @ self.M_hz.T / (self.hidden_dim if self.mode == 'cont' else 1.0)

            r = torch.sigmoid(ir + hr)
            z = torch.sigmoid(iz + hz)

            hh = ((r * x) @ self.N_hh) @ self.M_hh.T / (self.hidden_dim if self.mode == 'cont' else 1.0)
            g = self.nonlinearity(ih + hh)

            rec_x = (1.0 - z) * g + z * x

            # update hidden variable x accrding to the mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x

            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1, 0, 2)

        # get final output
        output = self.w_out(trajectories)

        return output, x, trajectories

    def clone(self):
        DEVICE = next(self.parameters()).device
        new_net = Low_Rank_GRU(self.input_size,self.output_size,self.hidden_dim,self.rank,self.nonlinearity,
                               self.mode,self.noise_std,self.tau,self.Win_bias, self.Wout_bias).to(DEVICE)

        new_net.M_hr = nn.Parameter(self.M_hr.detach().clone())
        new_net.N_hr = nn.Parameter(self.N_hr.detach().clone())

        new_net.M_hz = nn.Parameter(self.M_hz.detach().clone())
        new_net.N_hz = nn.Parameter(self.N_hz.detach().clone())

        new_net.M_hh = nn.Parameter(self.M_hh.detach().clone())
        new_net.N_hh = nn.Parameter(self.N_hh.detach().clone())

        new_net.W_ir = copy.deepcopy(self.W_ir)
        new_net.W_iz = copy.deepcopy(self.W_iz)
        new_net.W_ih = copy.deepcopy(self.W_ih)

        new_net.w_out = copy.deepcopy(self.w_out)
        return new_net


class HORNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity=torch.tanh, 
                 output_nonlinearity=torch.tanh, task="", mode="cont", form="rate",
                 noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, w_out = None, w_in = None,
                 adv=False,hard_orth=False):
        super(HORNN, self).__init__()

        self.hidden_dim=hidden_dim
        self.task=task
        self.adv = adv

        self.w_hh_rnn = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim))
        nn.init.normal_(self.w_hh_rnn, std=.1 / (hidden_dim**0.5))

        self.w_hh_tbrnn = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim,hidden_dim))
        nn.init.normal_(self.w_hh_tbrnn, std=.1 / (hidden_dim))

        if w_in is not None:
            self.w_in = w_in
        else:
            self.w_in = nn.Linear(input_size, hidden_dim, bias = Win_bias)
            #nn.init.xavier_uniform_(self.w_in.weight)
            nn.init.normal_(self.w_in.weight)
            if Win_bias is not None:
                nn.init.zeros_(self.w_in.bias)

        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, output_size, bias = Wout_bias)
            #nn.init.xavier_uniform_(self.w_out.weight)
            nn.init.normal_(self.w_out.weight,std=4.)


        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity


        self.task = None

        if mode not in ['cont', 'disc']:
            raise Exception("Error: Mode does not exists.")
        self.mode = mode

        if form not in ['rate', 'voltage']:
            raise Exception("Error: Form does not exists.")
        self.form = form

        self.input_size = input_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t, u_t in enumerate(u.permute(1, 0, 2)):
            if self.form == 'voltage':
                r = x
            elif self.form == 'rate':
                r = self.nonlinearity(x)

            # calculating: input weight
            input_I = self.w_in(u_t)

            hidden_W1 =  r @ self.w_hh_rnn.T 
            hidden_W2 = (r.view(b_size,1,1,hid) @ self.w_hh_tbrnn.view(1,hid,hid,hid) @ r.view(b_size,1,hid,1)).reshape((b_size,hid))

            if self.adv: # if setup is adversarial
                hidden_W = torch.sigmoid(self.alpha) * hidden_W1 + (1-torch.sigmoid(self.alpha)) * hidden_W2
            else:
                hidden_W = hidden_W1 + hidden_W2

            if self.form == 'voltage':
                rec_x = self.nonlinearity(hidden_W + input_I)
            elif self.form == 'rate':
                rec_x = hidden_W + input_I

            # update hidden variable x accrding to the currect mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x

            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1, 0, 2)

        # get final output
        if self.form == 'voltage':
            output = self.w_out(trajectories)
        elif self.form == 'rate':
            output = self.w_out(self.output_nonlinearity(trajectories))

        return output, x, trajectories



class Low_Rank_HORNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank_rnn=1, rank_tbrnn=1, nonlinearity=torch.tanh, 
                 output_nonlinearity=torch.tanh, task="", mode="cont", form="rate",
                 noise_std=5e-2, tau=0.2, Win_bias=True, Wout_bias=True, w_out = None, w_in = None,
                 adv=False,hard_orth=False):
        super(Low_Rank_HORNN, self).__init__()

        self.hidden_dim=hidden_dim
        self.task=task
        self.adv = adv

        # define a low rank RNN
        self.M_rnn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank_rnn)))
        self.N_rnn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank_rnn)))

        self.L_tbrnn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank_tbrnn)))
        self.M_tbrnn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank_tbrnn)))
        self.N_tbrnn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim,rank_tbrnn)))

        if w_in is not None:
            self.w_in = w_in
        else:
            self.w_in = nn.Linear(input_size, hidden_dim, bias = Win_bias)
            #nn.init.xavier_uniform_(self.w_in.weight)
            nn.init.normal_(self.w_in.weight)
            if Win_bias is not None:
                nn.init.zeros_(self.w_in.bias)

        if w_out is not None:
            self.w_out = w_out
        else:
            self.w_out = nn.Linear(hidden_dim, output_size, bias = Wout_bias)
            #nn.init.xavier_uniform_(self.w_out.weight)
            nn.init.normal_(self.w_out.weight,std=4.)


        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity


        self.task = None

        if mode not in ['cont', 'disc']:
            raise Exception("Error: Mode does not exists.")
        self.mode = mode

        if form not in ['rate', 'voltage']:
            raise Exception("Error: Form does not exists.")
        self.form = form

        self.input_size = input_size
        self.output_size = output_size
        self.rank_rnn = rank_rnn
        self.rank_tbrnn = rank_tbrnn
        self.noise_std = noise_std
        self.tau = tau
        self.Win_bias = Win_bias
        self.Wout_bias = Wout_bias
        self.hard_orth = hard_orth
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        DEVICE = next(self.parameters()).device
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length, b_size, hid, device=DEVICE)

        trajectories = torch.empty((seq_length, b_size, hid), device=DEVICE)
        if x0 is None:
            x0 = torch.zeros((b_size, hid), device=DEVICE)
        x = x0

        for t, u_t in enumerate(u.permute(1, 0, 2)):
            if self.form == 'voltage':
                r = x
            elif self.form == 'rate':
                r = self.nonlinearity(x)

            # calculating: input weight
            input_I = self.w_in(u_t)

            hidden_W_rnn = r @ self.N_rnn @ self.M_rnn.T / (self.hidden_dim if self.mode == 'cont' else 1.0)
            hidden_W_tbrnn = ((r @ self.M_tbrnn) * (r @ self.N_tbrnn)) @ self.L_tbrnn.T / (self.hidden_dim**2 if self.mode == 'cont' else 1.0)

            if self.adv: # if setup is adversarial
                hidden_W = torch.sigmoid(self.alpha) * hidden_W_rnn + (1-torch.sigmoid(self.alpha)) * hidden_W_tbrnn
            else:
                hidden_W = hidden_W_rnn + hidden_W_tbrnn

            if self.form == 'voltage':
                rec_x = self.nonlinearity(hidden_W + input_I)
            elif self.form == 'rate':
                rec_x = hidden_W + input_I

            # update hidden variable x accrding to the currect mode
            if self.mode == 'cont':
                x = x + self.noise_std * noise[t] + self.tau * (-x + rec_x)
            elif self.mode == 'disc':
                x = rec_x

            trajectories[t] = x

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1, 0, 2)

        # get final output
        if self.form == 'voltage':
            output = self.w_out(trajectories)
        elif self.form == 'rate':
            output = self.w_out(self.output_nonlinearity(trajectories))

        return output, x, trajectories

    def lr_to_tensor(self):
        M = self.M_rnn.T.cpu().detach()
        N = self.N_rnn.T.cpu().detach()
        W_rnn = torch.zeros(self.hidden_dim, self.hidden_dim)
        for m, n in zip(M, N):
            W_rnn += torch.outer(m, n)

        L = self.L_tbrnn.T.cpu().detach()
        M = self.M_tbrnn.T.cpu().detach()
        N = self.N_tbrnn.T.cpu().detach()
        W_tbrnn = torch.zeros(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        for l, m, n in zip(L, M, N):
            W_tbrnn += (l.view(self.hidden_dim, 1, 1) * torch.outer(m, n).unsqueeze(0))

        return W_rnn / self.hidden_dim, W_tbrnn / (self.hidden_dim ** 2)

    def clone(self):
        DEVICE = next(self.parameters()).device
        new_net = Low_Rank_HORNN(self.input_size, self.output_size, self.hidden_dim, self.rank_rnn, self.rank_tbrnn, self.nonlinearity, self.output_nonlinearity,
                                 self.task, self.mode, self.form, self.noise_std, self.tau, self.Win_bias, self.Wout_bias, adv=self.adv).to(DEVICE)
        new_net.L_tbrnn = nn.Parameter(self.L_tbrnn.detach().clone())
        new_net.M_tbrnn = nn.Parameter(self.M_tbrnn.detach().clone())
        new_net.N_tbrnn = nn.Parameter(self.N_tbrnn.detach().clone())
        new_net.M_rnn = nn.Parameter(self.M_rnn.detach().clone())
        new_net.N_rnn = nn.Parameter(self.N_rnn.detach().clone())
        new_net.w_in = copy.deepcopy(self.w_in)
        new_net.w_out = copy.deepcopy(self.w_out)
        return new_net