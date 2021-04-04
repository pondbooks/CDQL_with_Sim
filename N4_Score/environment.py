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

                #======================Hyper Parameter=====================
                weight = np.array([[1/4,1/4,1/4,1/4]])
                learn_alpha = 5e-5
                gamma = 0.99
                MAX_STEP = 1000
                #==========================================================

                state = np.array([[np.pi, 0.0]])            
                rewards = 0 #sum of rewards for each system (a,b)

                for test_step in range(MAX_STEP):
                    current_obs = torch.Tensor([[state[0,0],state[0,1]]])
                    action = self.agent.get_action(current_obs, weight) 
                    action = action.detach().numpy()[0] #action: torch.Tensor(1,1) -> numpy(1,)

                    #exploration noise=========================
                    noise = max((400-test_step),0.0)/400*0.1*np.random.normal()
                    action = action + noise #numpy(1,)
                    #==========================================

                    action = torch.Tensor([action]) #action: numpy(1,) -> torch.Tensor(1,1)
                    Q_vec = self.agent.get_Q_value(current_obs, action) # Q(x[k],a[k]) as characteristic functions: numpy(1,4)
                    action = action.detach().numpy()[0] #action: torch.Tensor(1,1) -> numpy(1,)

                    next_state, reward, done = dynamics.Dynamics(state, action, a_param, b_param)  #next_state: numpy(1,2)
                    next_obs = torch.Tensor([[next_state[0,0], next_state[0,1]]]) # numpy(1,2) -> torch(1,2)

                    #update of the parameters 
                    max_Q_next_vec = self.agent.get_next_value(next_obs, weight)

                    param = np.array([[weight[0,0], weight[0,1], weight[0,2], weight[0,3]]]) #w=[w_{1},...,w_{N}]
                    td_error = param @ Q_vec.T - (reward + gamma * (param @ max_Q_next_vec.T))

                    chara_vec = np.array([[Q_vec[0,0], Q_vec[0,1], Q_vec[0,2], Q_vec[0,3]]])

                    update_vec = td_error * chara_vec

                    #Barrier
                    eta = 1e-7
                    epsilon_w = 1e-9
                    barrier_vec = eta*np.array([[-1/(weight[0,0]+epsilon_w),\
                                                -1/(weight[0,1]+epsilon_w),\
                                                -1/(weight[0,2]+epsilon_w),\
                                                -1/(weight[0,3]+epsilon_w)]])

                    update_vec = update_vec + barrier_vec
                        
                    pre_weight = weight #memorize pre_weight
                    weight = weight - learn_alpha * (update_vec)          

                    if (weight[0,0]<0.0)or(weight[0,1]<0.0)or(weight[0,2]<0.0)or(weight[0,3]<0.0): #If some weights are negative 
                        update_error_count = 1
                        while(True):
                            weight = pre_weight 
                            weight = weight - (2**(-update_error_count))*learn_alpha * (update_vec)
                            update_error_count += 1
                            if (weight[0,0]>=0.0)and(weight[0,1]>=0.0)and(weight[0,2]>=0.0)and(weight[0,3]>=0.0):
                                break

                    #Normalize weight
                    weight_sum = weight[0,0] + weight[0,1] + weight[0,2] + weight[0,3]
                    weight = weight/weight_sum   
                    
                    rewards += reward
                    state = next_state

                  
                print("##########################################################################################")
                print("(a,b)=("+str(a_param)+","+str(b_param)+") : reward "+str(rewards)+" Weight is "+str(weight))
                print("Last State is "+str(state))
                print("##########################################################################################")
                vallist[b_iteration,a_iteration] = rewards

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["mathtext.fontset"] = 'cm'
        plt.rcParams['mathtext.default'] = 'it'

        fig, ax = plt.subplots()
        cs = ax.pcolormesh(xlist, ylist, vallist,  cmap="jet", vmin=-4000.0, vmax=-50.0)#seismic,hot
        fig.colorbar(cs)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(5.0, 50.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        ax.set_xlabel(r'$\xi_{1}$',fontsize=16)
        ax.set_ylabel(r'$\xi_{2}$',fontsize=16)
  
        fig.savefig('N4_case1.eps', pad_inches=0.05) 
        fig.savefig('N4_case1.png', pad_inches=0.05)
                
            
                
            
            