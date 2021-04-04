import torch
import matplotlib.pyplot as plt
import numpy as np

from agent import AGENT

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
        
        xlist = np.zeros((201,201))
        ylist = np.zeros((201,201))
        ulist = np.zeros((201,201))
           
        for episode in range(1):  

            for xx in range(201):
                for yy in range(201):
                    reset_x_1 = -np.pi + 0.0314*xx 
                    reset_x_2 = -4.0 + 0.04*yy 
            
                    plot_x = np.array([[reset_x_1, reset_x_2]])  # np.array(1,2)

                    xlist[xx,yy] = plot_x[0,0]
                    ylist[xx,yy] = plot_x[0,1]
                        
                    input_state = torch.zeros(1,3)
                    input_state[0,0] = np.sin(plot_x[0,0])
                    input_state[0,1] = np.cos(plot_x[0,0])
                    input_state[0,2] = plot_x[0,1]

                    u = self.agent.get_action(input_state,None)  


                    u = u.cpu() # Torch -> numpy
                    u = u.detach().numpy()
                    u = u.reshape(1,1) # numpy.array(1,1)

                    ulist[xx,yy] = u
                    

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams["mathtext.fontset"] = 'cm'
            plt.rcParams['mathtext.default'] = 'it'

            fig, ax = plt.subplots()
            cs = ax.pcolormesh(xlist, ylist, ulist, shading='auto', cmap='seismic', vmin=-1.0, vmax=1.0) # seismic,hot
            fig.colorbar(cs)

            # Plot cross mark
            ax.set_xlabel('$x_1$',fontsize=18)
            ax.set_ylabel('$x_2$',fontsize=18)
            point = {
                'fixedpoint':[0.0,0.0]
            }
            ax.plot(*point['fixedpoint'], 'x', color="gray", markersize=12)
         
            fig.savefig('mu_1.eps',  pad_inches=0.05) 
            fig.savefig('mu_1.png',  pad_inches=0.05) 
            
