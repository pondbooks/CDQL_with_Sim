# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:12:49 2020

@author: Junya
"""


from brain import BRAIN
import torch

class AGENT:
    def __init__(self, num_states, num_actions, args):
        self.args = args
        self.brain = BRAIN(num_states, num_actions, self.args)
        
    def update_DNNs(self, batch):
        self.brain.update_network(batch)
        
    def get_action(self, state, exploration_noise):
        action = self.brain.decide_action(state)
        #add noise############################################
        if exploration_noise is not None:
            action += torch.Tensor(exploration_noise.noise())  
            if action[0,0] > 1.0:
                action[0,0] = 1.0
            elif action[0,0] < -1.0:
                action[0,0] = -1.0
            else:
                action[0,0] = action[0,0]  
        ######################################################
        return action
    
    def update_target_DNNs(self):
        self.brain.update_target_network()
