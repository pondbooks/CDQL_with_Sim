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
        self.network_4 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        self.network_5 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        self.network_6 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        self.network_7 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)
        self.network_8 = network.NAF_Network(self.num_states,self.num_actions,self.args.hidden_layer_size)

        filename_1 ='weight_1.pth'
        filename_2 ='weight_2.pth'
        filename_3 ='weight_3.pth'
        filename_4 ='weight_4.pth'
        filename_5 ='weight_5.pth'
        filename_6 ='weight_6.pth'
        filename_7 ='weight_7.pth'
        filename_8 ='weight_8.pth'

        param_1 = torch.load(filename_1, map_location='cpu')
        param_2 = torch.load(filename_2, map_location='cpu')
        param_3 = torch.load(filename_3, map_location='cpu')
        param_4 = torch.load(filename_4, map_location='cpu')
        param_5 = torch.load(filename_5, map_location='cpu')
        param_6 = torch.load(filename_6, map_location='cpu')
        param_7 = torch.load(filename_7, map_location='cpu')
        param_8 = torch.load(filename_8, map_location='cpu')


        self.network_1.load_state_dict(param_1)
        self.network_2.load_state_dict(param_2)
        self.network_3.load_state_dict(param_3)
        self.network_4.load_state_dict(param_4)
        self.network_5.load_state_dict(param_5)
        self.network_6.load_state_dict(param_6)
        self.network_7.load_state_dict(param_7)
        self.network_8.load_state_dict(param_8)

        
    def decide_action(self, state, weight):
        self.network_1.eval()
        self.network_2.eval()
        self.network_3.eval()
        self.network_4.eval()
        self.network_5.eval()
        self.network_6.eval()
        self.network_7.eval()
        self.network_8.eval()


        weight_1 = weight[0,0]
        weight_2 = weight[0,1]
        weight_3 = weight[0,2]
        weight_4 = weight[0,3]
        weight_5 = weight[0,4]
        weight_6 = weight[0,5]
        weight_7 = weight[0,6]
        weight_8 = weight[0,7]


        state = torch.Tensor([[np.sin(state[0,0]), np.cos(state[0,0]), state[0,1]]])

        action_1, _, _, P_1 = self.network_1(state, None)
        action_2, _, _, P_2 = self.network_2(state, None)
        action_3, _, _, P_3 = self.network_3(state, None)
        action_4, _, _, P_4 = self.network_4(state, None)
        action_5, _, _, P_5 = self.network_5(state, None)
        action_6, _, _, P_6 = self.network_6(state, None)
        action_7, _, _, P_7 = self.network_7(state, None)
        action_8, _, _, P_8 = self.network_8(state, None)


        P_1 = P_1[0]
        P_2 = P_2[0]
        P_3 = P_3[0]
        P_4 = P_4[0]
        P_5 = P_5[0]
        P_6 = P_6[0]
        P_7 = P_7[0]
        P_8 = P_8[0]


        P_inv = (weight_1 * P_1 + weight_2 * P_2 + weight_3 * P_3 + weight_4 * P_4 + weight_5 * P_5\
                            + weight_6 * P_6 + weight_7 * P_7 + weight_8 * P_8)**(-1) #(1,1)

        action = (weight_1 * (P_1 @ action_1) + weight_2 * (P_2 @ action_2) + weight_3 * (P_3 @ action_3) +\
                 weight_4 * (P_4 @ action_4) + weight_5 * (P_5 @ action_5) + weight_6 * (P_6 @ action_6) +\
                 weight_7 * (P_7 @ action_7) + weight_8 * (P_8 @ action_8)) 

        action_ = P_inv @ action
        
        return action_

    def compute_Q_value(self, state, action):
        self.network_1.eval()
        self.network_2.eval()
        self.network_3.eval()
        self.network_4.eval()
        self.network_5.eval()
        self.network_6.eval()
        self.network_7.eval()
        self.network_8.eval()

        state = torch.Tensor([[np.sin(state[0,0]), np.cos(state[0,0]), state[0,1]]])

        _, Q_1, _, _ = self.network_1(state, action)
        _, Q_2, _, _ = self.network_2(state, action)
        _, Q_3, _, _ = self.network_3(state, action)
        _, Q_4, _, _ = self.network_4(state, action)
        _, Q_5, _, _ = self.network_5(state, action)
        _, Q_6, _, _ = self.network_6(state, action)
        _, Q_7, _, _ = self.network_7(state, action)
        _, Q_8, _, _ = self.network_8(state, action)

        Q_1 = Q_1.detach().numpy()[0,0]
        Q_2 = Q_2.detach().numpy()[0,0]
        Q_3 = Q_3.detach().numpy()[0,0]
        Q_4 = Q_4.detach().numpy()[0,0]
        Q_5 = Q_5.detach().numpy()[0,0]
        Q_6 = Q_6.detach().numpy()[0,0]
        Q_7 = Q_7.detach().numpy()[0,0]
        Q_8 = Q_8.detach().numpy()[0,0]

        Q_vector = np.array([[Q_1, Q_2, Q_3, Q_4, Q_5, Q_6, Q_7, Q_8]])

        return  Q_vector

    def compute_next_value(self, next_state, weight):
        self.network_1.eval()
        self.network_2.eval()
        self.network_3.eval()
        self.network_4.eval()
        self.network_5.eval()
        self.network_6.eval()
        self.network_7.eval()
        self.network_8.eval()

        weight_1 = weight[0,0]
        weight_2 = weight[0,1]
        weight_3 = weight[0,2]
        weight_4 = weight[0,3]
        weight_5 = weight[0,4]
        weight_6 = weight[0,5]
        weight_7 = weight[0,6]
        weight_8 = weight[0,7]

        next_state = torch.Tensor([[np.sin(next_state[0,0]), np.cos(next_state[0,0]), next_state[0,1]]])

        action_1, _, _, P_1 = self.network_1(next_state, None)
        action_2, _, _, P_2 = self.network_2(next_state, None)
        action_3, _, _, P_3 = self.network_3(next_state, None)
        action_4, _, _, P_4 = self.network_4(next_state, None)
        action_5, _, _, P_5 = self.network_5(next_state, None)
        action_6, _, _, P_6 = self.network_6(next_state, None)
        action_7, _, _, P_7 = self.network_7(next_state, None)
        action_8, _, _, P_8 = self.network_8(next_state, None)

        P_1 = P_1[0]
        P_2 = P_2[0]
        P_3 = P_3[0]
        P_4 = P_4[0]
        P_5 = P_5[0]
        P_6 = P_6[0]
        P_7 = P_7[0]
        P_8 = P_8[0]


        P_inv = (weight_1 * P_1 + weight_2 * P_2 + weight_3 * P_3 + weight_4 * P_4 + weight_5 * P_5\
                + weight_6 * P_6 + weight_7 * P_7 + weight_8 * P_8)**(-1) #(1,1)

        action = (weight_1 * (P_1 @ action_1) + weight_2 * (P_2 @ action_2) + weight_3 * (P_3 @ action_3)\
                + weight_4 * (P_4 @ action_4) + weight_5 * (P_5 @ action_5) + weight_6 * (P_6 @ action_6)\
                + weight_7 * (P_7 @ action_7) + weight_8 * (P_8 @ action_8)) 

        opt_action = P_inv @ action

        _, Q_1, _, _ = self.network_1(next_state, opt_action)
        _, Q_2, _, _ = self.network_2(next_state, opt_action)
        _, Q_3, _, _ = self.network_3(next_state, opt_action)
        _, Q_4, _, _ = self.network_4(next_state, opt_action)
        _, Q_5, _, _ = self.network_5(next_state, opt_action)
        _, Q_6, _, _ = self.network_6(next_state, opt_action)
        _, Q_7, _, _ = self.network_7(next_state, opt_action)
        _, Q_8, _, _ = self.network_8(next_state, opt_action)

        Q_1 = Q_1.detach().numpy()[0,0]
        Q_2 = Q_2.detach().numpy()[0,0]
        Q_3 = Q_3.detach().numpy()[0,0]
        Q_4 = Q_4.detach().numpy()[0,0]
        Q_5 = Q_5.detach().numpy()[0,0]
        Q_6 = Q_6.detach().numpy()[0,0]
        Q_7 = Q_7.detach().numpy()[0,0]
        Q_8 = Q_8.detach().numpy()[0,0]

        Q_next_vector = np.array([[Q_1, Q_2, Q_3, Q_4, Q_5, Q_6, Q_7, Q_8]])

        return  Q_next_vector
        
  