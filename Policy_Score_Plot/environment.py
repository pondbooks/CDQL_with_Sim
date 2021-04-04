import torch
import matplotlib.pyplot as plt
import numpy as np

from agent import AGENT
import dynamics


class ENVIRONMENT:
    
    def __init__(self, args, Ite):
        
        self.args = args
        self.Ite = Ite
        
        #Dim of state and action
        self.num_states = 3
        self.num_actions = 1
        
        #Initialize Agent
        self.agent = AGENT(self.num_states, self.num_actions, self.args)
        
    def run(self):
        
        xlist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ylist = np.array([5.0, 6.0, 7.0, 8.0, 9.0,\
                         10.0, 11.0, 12.0, 13.0, 14.0,\
                         15.0, 16.0, 17.0, 18.0, 19.0, \
                         20.0, 21.0, 22.0, 23.0, 24.0, \
                         25.0, 26.0, 27.0, 28.0, 29.0, \
                         30.0, 31.0, 32.0, 33.0, 34.0, \
                         35.0, 36.0, 37.0, 38.0, 39.0, \
                         40.0, 41.0, 42.0, 43.0, 44.0, \
                         45.0, 46.0, 47.0, 48.0, 49.0, 50.0])
        vallist = np.zeros((45,10))

        for a_iteration in range(10):
            a_param = 0.05 + a_iteration*0.1
            for b_iteration in range(45):
                b_param = 5.5 + b_iteration*1.0

                max_reward = -10000.0
                    
                # (a,b)'s score           
                rewards = 0

                theta = np.pi
                omega = 0.0
                state = np.array([[theta, omega]])

                        
                for test_step in range(1000):

                    current_obs = torch.Tensor([[np.sin(state[0,0]), np.cos(state[0,0]), state[0,1]]]) #state: numpy(1,2) -> torch.Tensor(1,3)
                    action = self.agent.get_action(current_obs, None) #Not input ounoise
                    action = action.detach().numpy()[0] #action: torch.Tensor(1,1) -> numpy(scaler)

                    next_state, reward, done = dynamics.Dynamics(state, action, a_param, b_param)  #next_state: numpy(1,2)
                        
                    rewards += reward
                        
                    state = next_state
                
                print("(a,b)=("+str(a_param)+","+str(b_param)+") : reward "+str(rewards))
                vallist[b_iteration,a_iteration] = rewards
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["mathtext.fontset"] = 'cm'
        plt.rcParams['mathtext.default'] = 'it' 
          
        fig, ax = plt.subplots()
        cs = ax.pcolormesh(xlist, ylist, vallist, cmap="jet", vmin=-4000.0, vmax=-50.0)#seismic,hot
        fig.colorbar(cs)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(5.0, 50.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        ax.set_xlabel(r'$\xi_{1}$',fontsize=18)
        ax.set_ylabel(r'$\xi_{2}$',fontsize=18)
  
        fig.savefig('Score_of_mu_1.eps', pad_inches=0.05) 
        fig.savefig('Score_of_mu_1.png', pad_inches=0.05)

            
                
            
            