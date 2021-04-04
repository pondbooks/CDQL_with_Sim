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
        
        self.main_network = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        
        filename ='weight_1.pth'
        param = torch.load(filename, map_location='cpu')
        self.main_network.load_state_dict(param) 
        
   
    def decide_action(self, state):
        self.main_network.eval()
        
        action, _, _, _ = self.main_network(state, None)
        
        return action
        
  