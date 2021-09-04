# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:28:05 2020

@author: Junya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class NAF_Network(nn.Module): 
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=5e-2):
        super(NAF_Network, self).__init__()
        
        self.num_actions = num_actions
        num_outputs = num_actions
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)

        self.mu = nn.Linear(hidden_size, num_outputs)

        self.L = nn.Linear(hidden_size, num_outputs ** 2)

        self.init_weights(init_w)

        self.tril_mask = torch.tril(torch.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(
            torch.ones(num_outputs, num_outputs))).unsqueeze(0) 
    
    def init_weights(self, init_w):
        #self.linear1.weight.data.uniform_(-init_w, init_w)
        #self.linear2.weight.data.uniform_(-init_w, init_w)
        #self.linear3.weight.data.uniform_(-init_w, init_w)
        #self.linear4.weight.data.uniform_(-init_w, init_w)
        self.V.weight.data.uniform_(-init_w, init_w)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.L.weight.data.uniform_(-init_w, init_w)
        
        
    def forward(self, x, u):
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        P = None

        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + torch.exp(L) * self.diag_mask.expand_as(L)
                
            P = torch.bmm(L, L.transpose(2, 1))
            u_mu = (u - mu).unsqueeze(2)
            A = -0.5*torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V, P
        
    


