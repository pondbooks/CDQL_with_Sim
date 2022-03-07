from brain import BRAIN
import torch

class AGENT:
    def __init__(self, num_states, num_actions, args):
        self.args = args
        self.brain = BRAIN(num_states, num_actions, self.args)
        
    def get_action(self, state, weight):
        action = self.brain.decide_action(state, weight)   
        return action

    def get_Q_value(self, state, action):
        Q_vector = self.brain.compute_Q_value(state, action) 
        return Q_vector
    
    def get_next_value(self, next_state, weight):
        V_next_vector = self.brain.compute_next_value(next_state, weight)
        return V_next_vector
    