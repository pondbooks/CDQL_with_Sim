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
        #episode_final = False
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["mathtext.fontset"] = 'cm'
        plt.rcParams['mathtext.default'] = 'it'
        params = {'legend.fontsize': 12,
          'legend.handlelength': 3}
        plt.rcParams.update(params)
        fig,axes = plt.subplots(nrows=5,ncols=1,figsize=(9,10))
        plt.subplots_adjust(hspace=0.8)
        #reward list
        sum_reward_list = []

        #======================Hyper Parameter=====================
        weight = np.array([[1/3,1/3,1/3]])
        learn_alpha = 5e-5
        gamma = 0.99
        MAX_STEP = 1000
        #==========================================================

        max_reward = -10000.0
 
        a_param = self.args.a_param
        b_param = self.args.b_param

        weight_1_list = []
        weight_2_list = []
        weight_3_list = []

        time_list = []
        a_list = []
        x_1_list = []
        x_2_list = []

        td_error_list = []

        Discrete_time = 0

        state = np.array([[np.pi, 0.0]])

        for test_step in range(MAX_STEP):

            #gradually change
            if test_step < 200:
                #a_param -= 0.9/200
                b_param -= 45.0/200

            weight_1_list.append(weight[0,0]) #store the initial parameter 
            weight_2_list.append(weight[0,1])
            weight_3_list.append(weight[0,2])

            time_list.append(test_step)
            x_1_list.append(state[0,0])
            x_2_list.append(state[0,1])

            current_obs = torch.Tensor([[state[0,0],state[0,1]]])
            action = self.agent.get_action(current_obs, weight) #Not input ounoise
            action = action.detach().numpy()[0] #action: torch.Tensor(1,1) -> numpy(scaler)

            #exploration noise###############################################
            if np.sqrt(state[0,0]**(2)+state[0,1]**(2))>=0.05: 
                noise = 0.1*np.random.normal()

            action = action + noise
            a_list.append(action)

            action = torch.Tensor([action])
            Q_vec = self.agent.get_Q_value(current_obs, action) # Q(x[k],a[k]) as characteristic functions
            action = action.detach().numpy()[0] #action: torch.Tensor(1,1) -> numpy(1,)

            next_state, reward, done = dynamics.Dynamics(state, action, a_param, b_param)  #next_state: numpy(1,2)
            next_obs = torch.Tensor([[next_state[0,0], next_state[0,1]]])

            #update of the parameters 
            max_Q_next_vec = self.agent.get_next_value(next_obs, weight)

            param = np.array([[weight[0,0], weight[0,1], weight[0,2]]]) #w=[w_{1},...,w_{N}]
            td_error = param @ Q_vec.T - (reward + gamma * (param @ max_Q_next_vec.T))
            td_error_list.append(abs(td_error[0,0]))
                
            chara_vec = np.array([[Q_vec[0,0], Q_vec[0,1], Q_vec[0,2]]])

            update_vec = td_error * chara_vec 

            #Barrier
            eta = 1e-7
            epsilon_w = 1e-9
            barrier_vec = eta*np.array([[-1/(weight[0,0]+epsilon_w),\
                                        -1/(weight[0,1]+epsilon_w),\
                                        -1/(weight[0,2]+epsilon_w)]])

            update_vec = update_vec + barrier_vec
                
            pre_weight = weight #memorize pre_weight
            weight = weight - learn_alpha * (update_vec) #weight is next weight

            if (weight[0,0]<0.0)or(weight[0,1]<0.0)or(weight[0,2]<0.0): #If some weights are negative 
                update_error_count = 1
                while(True):
                    weight = pre_weight 
                    weight = weight - (2**(-update_error_count))*learn_alpha * (update_vec)
                    update_error_count += 1
                    if (weight[0,0]>=0.0)and(weight[0,1]>=0.0)and(weight[0,2]>=0.0):
                        break
            weight_sum = weight[0,0]+weight[0,1]+weight[0,2]
            weight = weight/weight_sum

            if test_step%100==0:
                print(weight)
                print(str(a_param)+","+str(b_param))

            state = next_state
            Discrete_time += 1  

        axes[0].plot([0, MAX_STEP],[0, 0], "red", linestyle='dashed')
        axes[0].plot(time_list, a_list, linewidth=2)
        axes[0].set_xlim(0.0,MAX_STEP)
        axes[0].set_ylim(-1,1)
        axes[0].set_xlabel('$k$',fontsize=16)
        axes[0].set_ylabel('$a[k]$',fontsize=16)
        axes[0].grid(True)
            
        axes[1].plot([0, MAX_STEP],[0, 0], "red", linestyle='dashed')
        axes[1].plot(time_list, x_1_list, linewidth=2)
        axes[1].set_xlim(0.0,MAX_STEP)
        axes[1].set_ylim(-np.pi,np.pi)
        axes[1].set_xlabel('$k$',fontsize=16)
        axes[1].set_ylabel('$x_1[k]$',fontsize=16)
        axes[1].grid(True)
        
        axes[2].plot([0, MAX_STEP],[0, 0], "red", linestyle='dashed')
        axes[2].plot(time_list, x_2_list, linewidth=2)
        axes[2].set_xlim(0.0,MAX_STEP)
        axes[2].set_ylim(-7,7)
        axes[2].set_xlabel('$k$',fontsize=16)
        axes[2].set_ylabel('$x_2[k]$',fontsize=16)
        axes[2].grid(True) 

        axes[3].plot(time_list, weight_1_list, linewidth=2, label="$w_1$")
        axes[3].plot(time_list, weight_2_list, linewidth=2, label="$w_2$")
        axes[3].plot(time_list, weight_3_list, linewidth=2, label="$w_4$")
        axes[3].set_xlim(0.0,MAX_STEP)
        axes[3].set_ylim(0,1)
        axes[3].set_xlabel('$k$',fontsize=16)
        axes[3].set_ylabel('$w[k]$',fontsize=16)
        axes[3].grid(True) 
        axes[3].legend(loc='upper left',ncol=4)

        axes[4].plot(time_list, td_error_list, linewidth=2)
        axes[4].set_xlim(0.0,MAX_STEP)
       # axes[4].set_ylim(0,0.5)
        axes[4].set_xlabel('$k$',fontsize=16)
        axes[4].set_ylabel(r'$|\delta[k]|$',fontsize=16)
        axes[4].grid(True) 
        
        fig.savefig('from50_to5.eps', bbox_inches="tight", pad_inches=0.05) 
        fig.savefig('from50_to5.png', bbox_inches="tight", pad_inches=0.05) 
                
            
                
            
            