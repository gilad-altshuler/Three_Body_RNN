import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import sin_task
import K_Bit_Flipflop_task
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
          if hidden != None:
            # calculating: h^T @ W @ h + bias_tensor
            hidden_w = (hidden.view(b_size,1,1,hid) @ self.three_way_tensor.view(1,hid,hid,hid) @ hidden.view(b_size,1,hid,1)).reshape((b_size,hid))
            hidden_w+= self.bias_tensor

          # now perform the nonlinearity to the new hidden

          hidden = self.nonlinearity(hidden_w + input_w)
          r_out[i] = hidden

        # need r_out to be (batch_size, time_steps, hidden_size) dim:
        r_out = r_out.permute(1,0,2)

        # get final output
        output = self.fc(r_out)

        return output , hidden , r_out

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, hidden, _ = self.forward(X, None)

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


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())
        return losses


class Full_Rank_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity = 'tanh', task = "FF"):
        super(Full_Rank_RNN, self).__init__()

        self.hidden_dim=hidden_dim

        # define an RNN
        self.rnn = nn.RNN(input_size, hidden_dim, 1, nonlinearity=nonlinearity, batch_first=True)

        # fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.task = task

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

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            hidden = None
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, hidden, _ = self.forward(X, None)

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


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())
        return losses

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

    def lr_train(self,Inputs,tar,target_model,n_steps,optimizer,criterion,batch_size=128,T=100):
        Targets = target_model(Inputs,None)[2].detach().clone()
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                _, _, prediction = self.forward(X, None)
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

        with torch.no_grad():
            _,_,prediction = self.forward(Inputs, None)
            loss = criterion(prediction, Targets).cpu()
            print('Loss: ', loss.data.item())

        return losses

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, hidden, _ = self.forward(X, None)

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

        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())

        return losses


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

    def train(self,Inputs,Targets,n_steps,optimizer,criterion,batch_size=128,T=100):
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                prediction, hidden, _ = self.forward(X, None)

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


        with torch.no_grad():
          prediction, _ , _ = self.forward(Inputs, None)
          loss = criterion(prediction, Targets).cpu()
          print('Loss: ', loss.data.item())

        return losses

    def lr_train(self,Inputs,tar,target_model,n_steps,optimizer,criterion,batch_size=128,T=100):
        Targets = target_model(Inputs,None)[2].detach().clone()
        dataset = TensorDataset(Inputs, Targets)
        losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(n_steps)):
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            for X, Y in dataloader:
                optimizer.zero_grad()
                ####### outputs from the rnn ########
                _, _, prediction = self.forward(X, None)
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

        with torch.no_grad():
            _,_,prediction = self.forward(Inputs, None)
            loss = criterion(prediction, Targets).cpu()
            print('Loss: ', loss.data.item())

        return losses