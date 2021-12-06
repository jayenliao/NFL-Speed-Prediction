'''
TSA - NFL Project
Prdiction of Players' Speed
LSTM Experiments with Different Features

- Author: Jay C. Liao (jay-chiehen.liao@rennes-sb.com)
- Created on 4 Dec 2021
- Revised on 4 Dec 2021
- Finished on 5 Dec 2021
'''

import torch
from torch import nn
from tqdm import trange

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length, batch_size, device):
        super(myLSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers   # number of layers
        self.input_size = input_size   # input size
        self.hidden_size = hidden_size # hidden state
        self.seq_length = seq_length   # sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers) #lstm
        self.fc1 = nn.Linear(hidden_size, 10)  # fully connected 1
        self.fc2 = nn.Linear(10, batch_size)            # fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size) #internal state
        device = self.device
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x.to(device), (h_0.to(device), c_0.to(device))) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc1(out) #first Dense
        out = self.fc2(out)
        out = self.relu(out) #relu
        
        return out


def train(playId:int, model:myLSTM, epochs:int, loader_tr, optimizer, criterion, device):
    lst_loss = []
    pbar = trange(epochs)

    for epoch in pbar:   
        model.train()
        for x_batch, y_batch in loader_tr:
            outputs = model(x_batch) #forward pass
            m, n = outputs.shape
            outputs = outputs.reshape(n, m)
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch.float()).to(device)
            loss.backward() #calculates the loss of the loss function
            optimizer.step() #improve from loss, i.e backprop
        lst_loss.append(loss.item())
        pbar.set_description("Play %d Epoch=%s, MSE=%.4f" % (playId, epoch, loss.item()))

    return lst_loss