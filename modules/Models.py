import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import sin_task
import K_Bit_Flipflop_task
import PIN_task
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def correlation_regulation(V1,V2):
    if (V1==0).all() or (V2==0).all():
      return 0
    corr_matrix = (V1.t() @ V2).abs() / (V1.norm(dim=0,keepdim=True).t() @ V2.norm(dim=0,keepdim=True))
    return corr_matrix.mean()

def gram_schmidt(vecs,check_depend = True,epsilon=1e-5):
    vecs_orth = []
    vecs_orth.append(vecs[0] / vecs[0].norm())
    for i in range(1, len(vecs)):
        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]
            if check_depend and (v < epsilon).all().item(): # then its linear dependent set
                return []
        v = v / v.norm()
        vecs_orth.append(v)
    return vecs_orth

def project(vecs, subspace_vecs):
    """
    :param vecs: vectors to be projected
    :param subspace_vecs: the projecting space vectors
    :return: projected_vecs: corresponding projected vectors list
    """
    projected_vecs = []
    for v in vecs:
        subspace_orth = gram_schmidt(subspace_vecs,check_depend = False)
        M_proj = torch.stack(subspace_orth,dim=0)
        projected_vecs.append(M_proj.T @ M_proj @ v)
    return projected_vecs

class Three_Bodies_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity = torch.tanh, task = "FF"):
        super(Three_Bodies_RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity

        # define the input fc layer
        self.input_fc = nn.Linear(input_size, hidden_dim)
        nn.init.xavier_uniform_(self.input_fc.weight)       ###init
        self.input_fc.bias.data.fill_(0)                    ###init

        # define the hidden cross weight layer with bias
        self.three_way_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,hidden_dim,hidden_dim),requires_grad=True,device=DEVICE)))
        self.bias_tensor = nn.Parameter(torch.zeros(hidden_dim,requires_grad=True,device=DEVICE))

        # output fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, u, hidden):
        # u (batch_size, seq_length, input_size)
        # hidden (batch_size, hidden_dim)
        # r_out (batch_size, time_steps, hidden_size)
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        r_out = torch.empty((seq_length,b_size,hid),device=DEVICE)

        for i,u_t in enumerate(u.permute(1,0,2)):
          # calculating: input weight
            input_w = self.input_fc(u_t)
            hidden_w = 0
            if hidden is not None:
                # calculating: h^T @ W @ h + bias_tensor
                hidden_w = (hidden.view(b_size,1,1,hid) @ self.three_way_tensor.view(1,hid,hid,hid) @ hidden.view(b_size,1,hid,1)).reshape((b_size,hid))
                hidden_w += self.bias_tensor

            # now perform the nonlinearity to the new hidden

            hidden = self.nonlinearity(hidden_w + input_w)
            r_out[i] = hidden

        # need r_out to be (batch_size, time_steps, hidden_size) dim:
        r_out = r_out.permute(1,0,2)

        # get final output
        output = self.fc(r_out)
        return output, hidden, r_out

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,pin_task=False):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        hidden = None
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                if pin_task:
                    hidden = Y[:,0,:]

                prediction, _, _ = self.forward(X, hidden)

                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)
                loss = criterion(prediction, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps//50) == 0:
                print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task == "FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())
        return losses

    def evaluate(self,Inputs,Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs,None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def clone(self):
        new_net = Three_Bodies_RNN(self.input_size,self.output_size,self.hidden_dim,self.nonlinearity,self.task).to(DEVICE)
        new_net.input_fc = copy.deepcopy(self.input_fc)
        new_net.three_way_tensor = nn.Parameter(self.three_way_tensor.detach().clone())
        new_net.bias_tensor = nn.Parameter(self.bias_tensor.detach().clone())
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

class Full_Rank_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity = 'tanh', task = "FF"):
        super(Full_Rank_RNN, self).__init__()

        self.hidden_dim=hidden_dim

        # define an RNN
        self.rnn = nn.RNN(input_size, hidden_dim, 1, nonlinearity=nonlinearity, batch_first=True)

        # fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)

        # get final output
        output = self.fc(r_out)
        return output , hidden , r_out

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, _, _ = self.forward(X, hidden)

                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)
                loss = criterion(prediction, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())
        return losses

    def evaluate(self, Inputs, Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs, None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def clone(self):
        new_net = Full_Rank_RNN(self.input_size,self.output_size,self.hidden_dim,self.nonlinearity,self.task).to(DEVICE)
        new_net.rnn = copy.deepcopy(self.rnn)
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

class Low_Rank_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity = torch.tanh, task = "FF"):
        super(Low_Rank_RNN, self).__init__()

        self.hidden_dim=hidden_dim

        # define an low rank RNN
        self.U = nn.Parameter(nn.init.xavier_uniform_(torch.empty((rank,hidden_dim),requires_grad=True,device=DEVICE)))
        self.V = nn.Parameter(nn.init.xavier_uniform_(torch.empty((rank,hidden_dim),requires_grad=True,device=DEVICE)))

        self.input_fc = nn.Linear(input_size,hidden_dim)

        self.nonlinearity = nonlinearity

        # fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank

    def forward(self, x, hidden):
        # u (batch_size, seq_length, input_size)
        # hidden (batch_size, hidden_dim)
        # r_out (batch_size, time_steps, hidden_size)
        b_size, seq_length = x.size(0), x.size(1)
        hid = self.hidden_dim
        r_out = torch.empty((seq_length,b_size,hid),device=DEVICE)

        for i,x_t in enumerate(x.permute(1,0,2)):
          # calculating: input weight
          input_w = self.input_fc(x_t)

          hidden_w = 0
          if hidden != None:
            # calculating: sum (U_r @ V_r) @ ht
            hidden_w = (hidden @ self.V.T) @ self.U

          # now perform the nonlinearity to the new hidden

          hidden = self.nonlinearity(hidden_w + input_w)
          r_out[i] = hidden

        # need r_out to be (batch_size, time_steps, hidden_size) dim:
        r_out = r_out.permute(1,0,2)

        # get final output
        output = self.fc(r_out)

        return output , hidden , r_out

    def lr_train(self,Inputs,tar,target_model,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None):
        Targets = target_model(Inputs,None)[2].detach().clone()
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                _, _, prediction = self.forward(X, hidden)
                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward(retain_graph=True)
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                self.fc = target_model.fc
                prediction,_ ,r_out = self.forward(Inputs, None)
                loss = criterion(r_out, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                print(int(epoch/(n_steps/10)), '/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data,best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.cpu(),tar.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,tar.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())

        with torch.no_grad():
            _,_,prediction = self.forward(Inputs, None)
            loss = criterion(prediction, Targets).cpu()
            print('Loss: ', loss.data.item())

        return losses

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, _, _ = self.forward(X, hidden)

                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)
                loss = criterion(prediction, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())

        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())

        return losses

    def evaluate(self, Inputs, Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs, None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def clone(self):
        new_net = Low_Rank_RNN(self.input_size,self.output_size,self.hidden_dim,self.rank,self.nonlinearity,self.task).to(DEVICE)
        new_net.U = nn.Parameter(self.U.detach().clone())
        new_net.V = nn.Parameter(self.V.detach().clone())
        new_net.input_fc = copy.deepcopy(self.input_fc)
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

class Low_Rank_Three_Way_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity = torch.tanh, task = "FF"):
        super(Low_Rank_Three_Way_RNN, self).__init__()

        self.hidden_dim=hidden_dim

        # define an low rank RNN
        self.U = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))
        self.V = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))

        self.input_fc = nn.Linear(input_size,hidden_dim)
        nn.init.xavier_uniform_(self.input_fc.weight)       ###init
        self.input_fc.bias.data.fill_(0)                    ###init

        self.nonlinearity = nonlinearity

        # fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank

    def forward(self, x, hidden):
        # u (batch_size, seq_length, input_size)
        # hidden (batch_size, hidden_dim)
        # r_out (batch_size, time_steps, hidden_size)
        b_size, seq_length = x.size(0), x.size(1)
        hid = self.hidden_dim
        r_out = torch.empty((seq_length,b_size,hid),device=DEVICE)

        for i,x_t in enumerate(x.permute(1,0,2)):
          # calculating: input weight
          input_w = self.input_fc(x_t)

          hidden_w = 0
          if hidden != None:
            # calculating: sum h_t^T @ (W_r @ U_r @ V_r) @ ht
            hidden_w = (hidden @ self.U) * (hidden @ self.V)
            hidden_w = hidden_w @ self.W.T

          # now perform the nonlinearity to the new hidden

          hidden = self.nonlinearity(hidden_w + input_w)
          r_out[i] = hidden

        # need r_out to be (batch_size, time_steps, hidden_size) dim:
        r_out = r_out.permute(1,0,2)

        # get final output
        output = self.fc(r_out)

        return output , hidden , r_out

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, _, _ = self.forward(X, hidden)

                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)
                loss = criterion(prediction, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())

        return losses

    def lr_train(self,Inputs,tar,target_model,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None):
        Targets = target_model(Inputs,None)[2].detach().clone()
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                _, _, prediction = self.forward(X, hidden)
                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward(retain_graph=True)
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                self.fc = target_model.fc
                prediction,_ ,r_out = self.forward(Inputs, None)
                loss = criterion(r_out, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                print(int(epoch/(n_steps/10)), '/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data,best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.cpu(),tar.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,tar.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())

        with torch.no_grad():
            _,_,prediction = self.forward(Inputs, None)
            loss = criterion(prediction, Targets).cpu()
            print('Loss: ', loss.data.item())

        return losses

    def evaluate(self, Inputs, Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs, None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def lr_to_tensor(self):
        return torch.stack([(lr_rnn.U[:, [i]].unsqueeze(2) * (lr_rnn.V[:, [i]] @ lr_rnn.W[:, [i]].T)) for i in range(3)]).sum(dim=0)

    def clone(self):
        new_net = Low_Rank_Three_Way_RNN(self.input_size,self.output_size,self.hidden_dim,self.rank,self.nonlinearity,self.task).to(DEVICE)
        new_net.U = nn.Parameter(self.U.detach().clone())
        new_net.V = nn.Parameter(self.V.detach().clone())
        new_net.W = nn.Parameter(self.W.detach().clone())
        new_net.input_fc = copy.deepcopy(self.input_fc)
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

    def normalize_tensor(self):
        normU, normV, normW = self.U.norm(dim=0), self.V.norm(dim=0), self.W.norm(dim=0)
        coef = (normU * normV * normW) ** (1 / 3)
        _, indices = coef.sort()
        self.U, self.V, self.W = nn.Parameter((self.U * coef / normU)[:, indices]), nn.Parameter(
            (self.V * coef / normV)[:, indices]), nn.Parameter((self.W * coef / normW)[:, indices])

    ######################## Models for dynamics ##################################

class Three_Bodies_RNN_Dynamics(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity=torch.tanh, task = "FF", noise_std=5e-2, tau=0.2):
        super(Three_Bodies_RNN_Dynamics, self).__init__()

        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity

        # define the input fc layer
        self.input_fc = nn.Linear(input_size, hidden_dim,bias=False)
        nn.init.xavier_uniform_(self.input_fc.weight)       ###init

        # define the hidden cross weight layer with bias
        self.three_way_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,hidden_dim,hidden_dim),requires_grad=True,device=DEVICE)))
        self.bias_tensor = nn.Parameter(torch.zeros(hidden_dim,requires_grad=True,device=DEVICE))

        # output fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size,bias=False)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.tau = tau

    def forward(self, u, hidden):
        # u (batch_size, seq_length, input_size)
        # hidden (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length,b_size,hid, device=DEVICE)

        trajectories = torch.empty((seq_length+1,b_size,hid),device=DEVICE)
        if hidden is None:
          hidden = torch.zeros((b_size, hid),device=DEVICE)
        trajectories[0] = hidden.clone()
        r = self.nonlinearity(hidden)

        for i,u_t in enumerate(u.permute(1,0,2)):
            # calculating: input weight
            input_I = self.input_fc(u_t)

            # calculating: h^T @ W @ h + bias_tensor
            hidden_W = (r.view(b_size,1,1,hid) @ self.three_way_tensor.view(1,hid,hid,hid) @ r.view(b_size,1,hid,1)).reshape((b_size,hid))

            # now perform the nonlinearity to the new hidden
            hidden+= self.noise_std * noise[i] + self.tau * (-hidden + hidden_W + input_I)

            trajectories[i+1] = hidden

            r = self.nonlinearity(hidden + self.bias_tensor)
        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1,0,2)

        # get final output
        output = self.fc(self.nonlinearity(trajectories[:,1:,:]))
        return output , hidden , trajectories

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,pin_task=False):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        hidden = None
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                if pin_task:
                    hidden = Y[:,0,:]

                prediction, _, _ = self.forward(X, hidden)

                ### calculate the loss
                loss = criterion(prediction, Y)
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)
                loss = criterion(prediction, Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps//50) == 0:
                print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task == "FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())
        return losses

    def evaluate(self,Inputs,Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs,None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def clone(self):
        new_net = Three_Bodies_RNN_Dynamics(self.input_size,self.output_size,self.hidden_dim,self.nonlinearity,self.task,self.noise_std,self.tau).to(DEVICE)
        new_net.input_fc = copy.deepcopy(self.input_fc)
        new_net.three_way_tensor = nn.Parameter(self.three_way_tensor.detach().clone())
        new_net.bias_tensor = nn.Parameter(self.bias_tensor.detach().clone())
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

class Low_Rank_Three_Way_RNN_Dynamics(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity = torch.tanh,
                 task = "FF",noise_std = 5e-2, tau=0.2,bias=True):
        super(Low_Rank_Three_Way_RNN_Dynamics, self).__init__()

        self.hidden_dim=hidden_dim

        # define a low rank RNN
        self.L = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))
        self.M = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))
        self.N = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))

        if bias:
            self.bias_tensor = nn.Parameter(torch.zeros(hidden_dim,requires_grad=True,device=DEVICE))
        else:
            self.bias_tensor = 0.0

        self.input_fc = nn.Linear(input_size,hidden_dim,bias=False)
        nn.init.xavier_uniform_(self.input_fc.weight)       ###init

        self.nonlinearity = nonlinearity

        # fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size,bias=False)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.noise_std = noise_std
        self.tau = tau
        self.bias=bias

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length,b_size,hid, device=DEVICE)

        trajectories = torch.empty((seq_length+1,b_size,hid),device=DEVICE)
        if x0 is None:
          x0 = torch.zeros((b_size, hid),device=DEVICE)
        trajectories[0] = x0.clone()
        x = x0
        r = self.nonlinearity(x0)

        for i,u_t in enumerate(u.permute(1,0,2)):
          # calculating: input weight
          input_I = self.input_fc(u_t)

          # calculating: sum h_t^T @ (1/N^2 * L_r @ M_r @ N_r) @ ht
          hidden_W = ((r @ self.M) * (r @ self.N)) @ self.L.T / (self.hidden_dim ** 2)

          # now perform the nonlinearity to the new hidden
          x+= self.noise_std * noise[i] + self.tau * (-x + hidden_W + input_I)

          trajectories[i+1] = x

          r = self.nonlinearity(x + self.bias_tensor)

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1,0,2)

        # get final output
        output = self.fc(self.nonlinearity(trajectories[:,1:,:]))

        return output , x , trajectories

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None,reg=False):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, _, _ = self.forward(X, hidden)

                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.L, self.input_fc.weight)

                ### calculate the loss
                loss = criterion(prediction, Y) + regulation
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)

                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.L, self.input_fc.weight)

                loss = criterion(prediction, Targets) + regulation
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                if reg:
                    print(int(epoch / (n_steps / 10)),
                          '/10 --- loss = {:.6f},reg = {:.6f} best = {:.6f}'.format(loss.cpu().data,
                                                                                    regulation.cpu().data,
                                                                                    best_loss.cpu().data))
                else:
                    print(int(epoch / (n_steps / 10)),
                          '/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)

          regulation = 0
          if reg:
              regulation = correlation_regulation(self.L, self.input_fc.weight)

          loss = criterion(prediction, Targets).cpu() + regulation
          print('Loss: ', loss.data.item())

        return losses

    def lr_train(self,Inputs,tar,target_model,n_steps,optimizer,
                 criterion,batch_size=128,T=100,hidden=None,reg=False):
        Targets = self.nonlinearity(target_model(Inputs,None)[2].detach().clone())
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                _, _, trajectories = self.forward(X, hidden)
                prediction = self.nonlinearity(trajectories)
                ### calculate the loss
                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.L, self.input_fc.weight)
                loss = criterion(prediction, Y) + regulation
                ### perform backprop and update weights
                loss.backward(retain_graph=True)
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                self.fc = target_model.fc
                prediction,_ ,trajectories = self.forward(Inputs, None)
                trajectories = self.nonlinearity(trajectories)

                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.L, self.input_fc.weight)

                loss = criterion(trajectories,Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                if reg:
                    print(int(epoch/(n_steps/10)), '/10 --- loss = {:.6f},reg = {:.6f} best = {:.6f}'.format(loss.cpu().data,regulation.cpu().data,best_loss.cpu().data))
                else:
                    print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data,best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.cpu(),tar.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,tar.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())

        with torch.no_grad():
            _,_,trajectories = self.forward(Inputs, None)
            prediction = self.nonlinearity(trajectories)

            regulation = 0
            if reg:
                regulation = correlation_regulation(self.L, self.input_fc.weight)

            loss = criterion(prediction, Targets).cpu() + regulation
            print('Loss: ', loss.data.item())

        return losses

    def evaluate(self, Inputs, Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs, None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def lr_to_tensor(self):
        L = self.L.T.cpu().detach()
        M = self.M.T.cpu().detach()
        N = self.N.T.cpu().detach()
        W = torch.zeros(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        for l, m, n in zip(L, M, N):
            W += (l.view(self.hidden_dim, 1, 1) * torch.outer(m, n).unsqueeze(0))
        return W / (self.hidden_dim ** 2)

    def clone(self):
        new_net = Low_Rank_Three_Way_RNN_Dynamics(self.input_size,self.output_size,self.hidden_dim,self.rank,self.nonlinearity,self.task,self.noise_std,self.tau,self.bias).to(DEVICE)
        new_net.L = nn.Parameter(self.L.detach().clone())
        new_net.M = nn.Parameter(self.M.detach().clone())
        new_net.N = nn.Parameter(self.N.detach().clone())
        if self.bias:
            new_net.bias_tensor = nn.Parameter(self.bias_tensor.detach().clone())
        else:
            new_net.bias_tensor = 0.0
        new_net.input_fc = copy.deepcopy(self.input_fc)
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

    def normalize_tensor_(self): #inplace function

        # first, check that vecs are independent in each L,M,N matrices
        from utils import gram_schmidt
        if not gram_schmidt(self.L.T.cpu().detach().clone().numpy()):
            raise Exception(f"L vectors are linear dependant, cannot normalize")
        if not gram_schmidt(self.M.T.cpu().detach().clone().numpy()):
            raise Exception(f"M vectors are linear dependant, cannot normalize")
        if not gram_schmidt(self.N.T.cpu().detach().clone().numpy()):
            raise Exception(f"N vectors are linear dependant, cannot normalize")

        # Now make the tensor unique
        norml,normm,normn = self.L.norm(dim=0), self.M.norm(dim=0), self.N.norm(dim=0)
        coef = (norml * normm * normn)**(1/3)
        _,indices = coef.sort()
        self.L, self.M, self.N = nn.Parameter((self.L*coef/norml)[:,indices]), nn.Parameter((self.M*coef/normm)[:,indices]), nn.Parameter((self.N*coef/normn)[:,indices])

    def normalize_tensor(self):
        # first, check that vecs are independent in each L,M,N matrices
        from utils import gram_schmidt
        if not gram_schmidt(self.L.T.cpu().detach().clone().numpy()):
            raise Exception(f"L vectors are linear dependent, cannot normalize")
        if not gram_schmidt(self.M.T.cpu().detach().clone().numpy()):
            raise Exception(f"M vectors are linear dependent, cannot normalize")
        if not gram_schmidt(self.N.T.cpu().detach().clone().numpy()):
            raise Exception(f"N vectors are linear dependent, cannot normalize")

        # Now make the tensor unique
        norml, normm, normn = self.L.norm(dim=0), self.M.norm(dim=0), self.N.norm(dim=0)
        coef = (norml * normm * normn) ** (1 / 3)
        _, indices = coef.sort()
        return (self.L * coef / norml)[:, indices],(self.M * coef / normm)[:, indices],(self.N * coef / normn)[:, indices]

class Low_Rank_RNN_Dynamics(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, rank=1, nonlinearity = torch.tanh,
                 task = "FF",noise_std = 5e-2,tau=0.2,bias=True):
        super(Low_Rank_RNN_Dynamics, self).__init__()

        self.hidden_dim=hidden_dim

        # define a low rank RNN
        self.M = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))
        self.N = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim,rank),requires_grad=True,device=DEVICE)))
        if bias:
            self.bias_tensor = nn.Parameter(torch.zeros(hidden_dim,requires_grad=True,device=DEVICE))
        else:
            self.bias_tensor = 0.0

        self.input_fc = nn.Linear(input_size,hidden_dim,bias=False)
        nn.init.xavier_uniform_(self.input_fc.weight)       ###init

        self.nonlinearity = nonlinearity

        # fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size,bias=False)

        self.task = task
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.noise_std = noise_std
        self.tau = tau
        self.bias = bias

    def forward(self, u, x0):
        # u (batch_size, seq_length, input_size)
        # x0 (batch_size, hidden_dim)
        # trajectories (batch_size, time_steps, hidden_size)
        b_size, seq_length = u.size(0), u.size(1)
        hid = self.hidden_dim
        noise = torch.randn(seq_length,b_size,hid, device=DEVICE)

        trajectories = torch.empty((seq_length+1,b_size,hid),device=DEVICE)
        if x0 is None:
          x0 = torch.zeros((b_size, hid),device=DEVICE)
        trajectories[0] = x0.clone()
        x = x0
        r = self.nonlinearity(x0)

        for i,u_t in enumerate(u.permute(1,0,2)):
          # calculating: input weight
          input_I = self.input_fc(u_t)

          # calculating: sum (1/N * M_r @ N_r) @ rt
          hidden_W = r @ self.N @ self.M.T / self.hidden_dim

          # now perform the nonlinearity to the new hidden
          x+= self.noise_std * noise[i] + self.tau * (-x + hidden_W + input_I)

          trajectories[i+1] = x

          r = self.nonlinearity(x + self.bias_tensor)

        # need trajectories to be (batch_size, time_steps, hidden_size) dim:
        trajectories = trajectories.permute(1,0,2)

        # get final output
        output = self.fc(self.nonlinearity(trajectories[:,1:,:]))

        return output , x , trajectories

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None,reg=False):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, _, _ = self.forward(X, hidden)
                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.M, self.input_fc.weight)
                ### calculate the loss
                loss = criterion(prediction, Y) + regulation
                ### perform backprop and update weights
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                prediction,_ ,_ = self.forward(Inputs, None)
                loss = criterion(prediction, Targets)

                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.M, self.input_fc.weight)

                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                if reg:
                    print(int(epoch / (n_steps / 10)),'/10 --- loss = {:.6f},reg = {:.6f} best = {:.6f}'.format(loss.cpu().data,regulation.cpu().data,best_loss.cpu().data))
                else:
                    print(int(epoch / (n_steps / 10)),
                          '/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.to("cpu"),Targets.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,Targets.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          regulation = 0
          if reg:
              regulation = correlation_regulation(self.M, self.input_fc.weight)
          loss = criterion(prediction, Targets).cpu() + regulation
          print('Loss: ', loss.data.item())

        return losses

    def lr_train(self,Inputs,tar,target_model,n_steps,optimizer,criterion,batch_size=128,T=100,hidden=None,reg=False):
        Targets = self.nonlinearity(target_model(Inputs,None)[2].detach().clone())
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                _, _, trajectories = self.forward(X, hidden)
                prediction = self.nonlinearity(trajectories)

                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.M, self.input_fc.weight)

                ### calculate the loss
                loss = criterion(prediction, Y) + regulation
                ### perform backprop and update weights
                loss.backward(retain_graph=True)
                optimizer.step()

            torch.cuda.empty_cache()
            # display loss and predictions
            with torch.no_grad():
                self.fc = target_model.fc
                prediction,_ ,trajectories = self.forward(Inputs, None)
                trajectories = self.nonlinearity(trajectories)

                regulation = 0
                if reg:
                    regulation = correlation_regulation(self.M, self.input_fc.weight)

                loss = criterion(trajectories,Targets)
                losses.append(loss.cpu().data.item())
                if best_loss > loss:
                    self.best_model = copy.deepcopy(self.state_dict())
                    best_loss = loss

            if epoch % (n_steps/10) == 0:
                if reg:
                    print(int(epoch / (n_steps / 10)),
                          '/10 --- loss = {:.6f},reg = {:.6f} best = {:.6f}'.format(loss.cpu().data,
                                                                                    regulation.cpu().data,
                                                                                    best_loss.cpu().data))
                else:
                    print(int(epoch / (n_steps / 10)),
                          '/10 --- loss = {:.6f}, best = {:.6f}'.format(loss.cpu().data, best_loss.cpu().data))
                if self.task =="FF":
                    K_Bit_Flipflop_task.draw_mem_charts(Inputs.cpu(),tar.cpu(),Prediction=prediction.cpu(),idx=0,K=Inputs.shape[2])
                elif self.task == "Sin":
                    sin_task.plot_sin(T,tar.cpu(),prediction.cpu(),loss)
                elif self.task == "PIN":
                    PIN_task.plot_proteins(Targets[0].T.cpu(),prediction[0].T.cpu())

        with torch.no_grad():
            _,_,trajectories = self.forward(Inputs, None)
            prediction = self.nonlinearity(trajectories)
            regulation = 0
            if reg:
                regulation = correlation_regulation(self.M, self.input_fc.weight)
            loss = criterion(prediction, Targets).cpu() + regulation
            print('Loss: ', loss.data.item())

        return losses

    def evaluate(self, Inputs, Targets):
        with torch.no_grad():
            predictions = self.forward(Inputs, None)[0].sign()
            accuracy = (Targets == predictions).float().mean()
            return accuracy

    def lr_to_tensor(self):
        M = self.M.T.cpu().detach()
        N = self.N.T.cpu().detach()
        W = torch.zeros(self.hidden_dim, self.hidden_dim)
        for m, n in zip(M, N):
            W += torch.outer(m, n)
        return W / self.hidden_dim

    def clone(self):
        new_net = Low_Rank_RNN_Dynamics(self.input_size,self.output_size,self.hidden_dim,self.rank,self.nonlinearity,self.task,self.noise_std,self.tau,self.bias).to(DEVICE)
        new_net.M = nn.Parameter(self.M.detach().clone())
        new_net.N = nn.Parameter(self.N.detach().clone())
        if self.bias:
            new_net.bias_tensor = nn.Parameter(self.bias_tensor.detach().clone())
        else:
            new_net.bias_tensor = 0.0
        new_net.input_fc = copy.deepcopy(self.input_fc)
        new_net.fc = copy.deepcopy(self.fc)
        return new_net

    def normalize_tensor(self):
        with torch.no_grad():
            structure = (self.M @ self.N.t()).cpu().detach().numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            return torch.from_numpy(m * np.sqrt(s)).to(DEVICE),torch.from_numpy(n.transpose() * np.sqrt(s)).to(DEVICE)

    def normalize_tensor_(self):
        with torch.no_grad():
            structure = (self.M @ self.N.t()).cpu().detach().numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.M.set_(torch.from_numpy(m * np.sqrt(s)).to(DEVICE))
            self.N.set_(torch.from_numpy(n.transpose() * np.sqrt(s)).to(DEVICE))