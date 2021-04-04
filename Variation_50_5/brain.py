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
        
        self.network_1 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        self.network_2 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        self.network_3 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)

        filename_1 ='weight_1.pth'
        filename_2 ='weight_2.pth'
        filename_3 ='weight_4.pth'
        
        param_1 = torch.load(filename_1, map_location='cpu')
        param_2 = torch.load(filename_2, map_location='cpu')
        param_3 = torch.load(filename_3, map_location='cpu')

        self.network_1.load_state_dict(param_1)
        self.network_2.load_state_dict(param_2)
        self.network_3.load_state_dict(param_3)
        
    def decide_action(self, state, weight):
        self.network_1.eval()
        self.network_2.eval()
        self.network_3.eval()

        weight_1 = weight[0,0]
        weight_2 = weight[0,1]
        weight_3 = weight[0,2]

        state = torch.Tensor([[np.sin(state[0,0]), np.cos(state[0,0]), state[0,1]]])
        
        action_1, _, _, P_1 = self.network_1(state, None)
        action_2, _, _, P_2 = self.network_2(state, None)
        action_3, _, _, P_3 = self.network_3(state, None)

        P_1 = P_1[0]
        P_2 = P_2[0]
        P_3 = P_3[0]

        P_inv = (weight_1 * P_1 + weight_2 * P_2 + weight_3 * P_3)**(-1) #(1,1)

        action = (weight_1 * (P_1 @ action_1) + weight_2 * (P_2 @ action_2) + weight_3 * (P_3 @ action_3)) 

        action_ = P_inv @ action
        
        return action_

    def compute_Q_value(self, state, action):
        self.network_1.eval()
        self.network_2.eval()
        self.network_3.eval()

        state = torch.Tensor([[np.sin(state[0,0]), np.cos(state[0,0]), state[0,1]]])

        _, Q_1, _, _ = self.network_1(state, action)
        _, Q_2, _, _ = self.network_2(state, action)
        _, Q_3, _, _ = self.network_3(state, action)

        Q_1 = Q_1.detach().numpy()[0,0]
        Q_2 = Q_2.detach().numpy()[0,0]
        Q_3 = Q_3.detach().numpy()[0,0]

        Q_vector = np.array([[Q_1, Q_2, Q_3]])

        return  Q_vector

    def compute_next_value(self, next_state, weight):
        self.network_1.eval()
        self.network_2.eval()
        self.network_3.eval()

        weight_1 = weight[0,0]
        weight_2 = weight[0,1]
        weight_3 = weight[0,2]

        next_state = torch.Tensor([[np.sin(next_state[0,0]), np.cos(next_state[0,0]), next_state[0,1]]])

        action_1, _, _, P_1 = self.network_1(next_state, None)
        action_2, _, _, P_2 = self.network_2(next_state, None)
        action_3, _, _, P_3 = self.network_3(next_state, None)

        P_1 = P_1[0]
        P_2 = P_2[0]
        P_3 = P_3[0]

        P_inv = (weight_1 * P_1 + weight_2 * P_2 + weight_3 * P_3)**(-1) #(1,1)

        action = (weight_1 * (P_1 @ action_1) + weight_2 * (P_2 @ action_2) + weight_3 * (P_3 @ action_3)) 

        opt_action = P_inv @ action

        _, Q_1, _, _ = self.network_1(next_state, opt_action)
        _, Q_2, _, _ = self.network_2(next_state, opt_action)
        _, Q_3, _, _ = self.network_3(next_state, opt_action)

        Q_1 = Q_1.detach().numpy()[0,0]
        Q_2 = Q_2.detach().numpy()[0,0]
        Q_3 = Q_3.detach().numpy()[0,0]

        Q_next_vector = np.array([[Q_1, Q_2, Q_3]]) #(1,4)

        return  Q_next_vector
        
  