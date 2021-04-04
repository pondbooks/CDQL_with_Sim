from brain import BRAIN
import torch

class AGENT:
    def __init__(self, num_states, num_actions, args):
        self.args = args
        self.brain = BRAIN(num_states, num_actions, self.args)


    def get_action(self, state, exploration_noise):
        action = self.brain.decide_action(state)
        if exploration_noise is not None:
            action += torch.Tensor(exploration_noise.noise())     
        return action