#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:25:55 2019

@author: augusto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Una convolucion simple
        self.conv1 = nn.Conv1d(1, 1, 6)
        # La salida de la convolucion de arriba tiene
        # tamaño size_entrada - 6 + 1
        # y = w x + b
        self.lin1 = nn.Linear((24 - 6 + 1), 1)
        # Salida de 1 dimension
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lin1(x)
        
        return x
    

if __name__ == "__main__":
    total_epochs = 100
    # Definicion del modelo
    net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 1, 6),
            #torch.nn.BatchNorm1d(hidden_dim_1),
            #torch.nn.Conv1d(1, 1, hidden_dim_2),
            torch.nn.ReLU(),
            torch.nn.Linear((24 - 6 + 1), 10),
            torch.nn.Dropout(p = 0.2),
            torch.nn.Linear(10, 5),
            torch.nn.Dropout(p = 0.2),
            torch.nn.Linear(5, 1),
            torch.nn.Sigmoid()
            )
    
    aux_x = [1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3]
    
    x = torch.Tensor(aux_x)
    # Reshape al tamaño  correcto
    x = x.view(1, 1, 24)
    #x = torch.randn(1, 1, 24)
    target = torch.ones(1, 1, 1)
    
    learning_rate = 0.01
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    
    for this_epoch in range(total_epochs):
        out = net(x)
        
        loss = criterion(out,target)
        
        print("Epoch: {} - Loss:".format(this_epoch))
        print(loss)
        
        net.zero_grad()
        
        loss.backward()
        
        optimizer.step()