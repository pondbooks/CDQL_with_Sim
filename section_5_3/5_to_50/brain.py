# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:13:30 2020

@author: Junya
"""

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

import network

class BRAIN:
    def __init__(self, num_states, num_actions, args):
        self.args = args
        self.num_states = num_states
        self.num_actions = num_actions
        
        #Set 2 DNNs
        self.main_network = network.NAF_Network(self.num_states, self.num_actions, self.args.hidden_layer_size) 
        self.target_network = network.NAF_Network(self.num_states, self.num_actions, self.args.hidden_layer_size)  
        
        # initialize parameter vector
        filename ='weight_2.pth'
        param = torch.load(filename, map_location='cpu')
        self.main_network.load_state_dict(param)
        
        #Output information about Actor network and Critic network
        print(self.main_network)  
        
        self.target_network.load_state_dict(self.main_network.state_dict()) 
        
        #Set Optimizer
        self.optimizer = optim.Adam(
            self.main_network.parameters(), lr=self.args.lr)
        
        self.loss_func = nn.MSELoss()
        
    def update_network(self, batch):
        #Make minibatch
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        mask_batch = torch.cat(batch.mask)
        next_state_batch = torch.cat(batch.next_state)
        
        #Change eval mode
        self.main_network.eval()
        self.target_network.eval()
        
        #Compute Target Value t ###########################
        _, _, next_state_values, _ = self.target_network(next_state_batch, None)
        next_state_values = next_state_values.detach() #target value is constant.

        reward_batch_ = reward_batch.unsqueeze(1)
        mask_batch_ = mask_batch.unsqueeze(1)

        expected_Q_value_batch = reward_batch_ + (self.args.gamma * mask_batch_ * next_state_values) 
        #####################################################

        #Change train mode
        self.main_network.train()
        
        _, Q_value_batch, _, _ = self.main_network(state_batch, action_batch)
        
        loss = self.loss_func(Q_value_batch, expected_Q_value_batch) 
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm(self.main_network.parameters(),1)
        self.optimizer.step()

        
    def update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)

    def decide_action(self, state):
        self.main_network.eval()
        
        action, _, _, _ = self.main_network(state, None)
        
        return action
        
  