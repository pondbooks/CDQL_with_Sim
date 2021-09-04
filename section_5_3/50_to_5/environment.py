# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:11:40 2020

@author: Junya
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from agent import AGENT
from replay_memory import MEMORY, Transition
import dynamics

class ENVIRONMENT:
    
    def __init__(self, args, Ite):
        
        self.args = args
        self.Ite = Ite
        
        #Dim of state and action
        self.num_states = 3 #x=[sin\theta cos\theta \omega]
        self.num_actions = 1 #a=[a]
        
        #Initialize Agent
        self.agent = AGENT(self.num_states, self.num_actions, self.args)
        
    def run(self):        
        #Initialize Replay Memory (class)
        memory = MEMORY(self.args.replay_buffer_size)
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["mathtext.fontset"] = 'cm'
        plt.rcParams['mathtext.default'] = 'it'
        params = {'legend.fontsize': 12,
          'legend.handlelength': 3}
        plt.rcParams.update(params)

        fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(9,6))
        plt.subplots_adjust(hspace=0.5)

        sum_of_rewards = 0

        a_param = self.args.a_param
        b_param = self.args.b_param
           
        #Learning Phase
        state = dynamics.Initialize() # Get the initial state s_0 (numpy(1,2))
        print("Initial State is "+str(state))

        print('This episode mass:'+str(a_param))
        print('This episode length:'+str(b_param))

        MAX_STEP = 1000

        time_list = []
        a_list = []
        x_1_list = []
        x_2_list = []
            
        for learning_step in range(MAX_STEP): 
            #gradually change
            if learning_step < 200:
                b_param -= 45.0/200
            x_1_list.append(state[0,0])
            x_2_list.append(state[0,1])
            time_list.append(learning_step)

            current_obs = torch.Tensor([[np.sin(state[0,0]), np.cos(state[0,0]), state[0,1]]]) #state: numpy(1,2) -> torch.Tensor(1,3)
            action = self.agent.get_action(current_obs, None) #exploration action by agent (torch.Tensor(1,1))
            action = action.detach().numpy()[0] #action: torch.Tensor(1,1) -> numpy(1,)
            #exploration noise###############################################
            if np.sqrt(state[0,0]**(2)+state[0,1]**(2))>=0.05: 
                noise = 0.1*np.random.normal()
            action = action + noise
            a_list.append(action)

            next_state, reward, done = dynamics.Dynamics(state, action, a_param, b_param)  #next_state: numpy(1,2)
            sum_of_rewards += reward

            #Make Exploration
            action = torch.Tensor([action]) #action: numpy(1,) -> torch.Tensor(1,1)
            mask = torch.Tensor([not done]) #mask: bool(False) -> torch.Tensor(1)(True)
            next_obs = torch.Tensor([[np.sin(next_state[0,0]), np.cos(next_state[0,0]), next_state[0,1]]]) #next_state: numpy(1,2) -> torch.Tensor(1,3)
            reward = torch.Tensor([reward]) #reward: numpy(scalar) -> torch.Tensor(1)
                
            if abs(action[0])<=1.0: #If we do not want to store the experience that has the big scale action.
                memory.push(current_obs, action, mask, next_obs, reward) # all torch.Tensor
                
            state = next_state 
                
            #Update main DNN and target DNN
            if len(memory) > self.args.batch_size:
                transitions = memory.sample(self.args.batch_size) #Make exploration_batch
                batch = Transition(*zip(*transitions))
                    
                self.agent.update_DNNs(batch) #Update DNN
                self.agent.update_target_DNNs() #Update Target DNN
                
            #if done:
                #break
            
        print("Sum of rewards is "+str(sum_of_rewards))
            
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

        fig.savefig('standard_from50_to_5.eps', bbox_inches="tight", pad_inches=0.05) 
        fig.savefig('standard_from50_to_5.png', bbox_inches="tight", pad_inches=0.05) 
     
            