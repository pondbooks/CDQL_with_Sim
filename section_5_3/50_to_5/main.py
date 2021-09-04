# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:08:37 2020

@author: Junya
"""

import torch 
import numpy as np
import fixed_seed
import argparse

from environment import ENVIRONMENT

def main():
    #torch.utils.backcompat.broadcast_warning.enabled = True
    #torch.utils.backcompat.keepdim_warning.enabled = True

    #torch.set_default_tensor_type('torch.DoubleTensor')

    parser = argparse.ArgumentParser(description='PyTorch NAF-pendulum example')
    
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='soft update parameter (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='Batch size (default: 128)')
    parser.add_argument('--replay_buffer_size', type=int, default=1e6, metavar='N',
                        help='Replay Buffer Size (default: 1e6)')
    parser.add_argument('--hidden_layer_size', type=int, default=128, metavar='N',
                        help='Hidden Layer Size (default: 64)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='G',
                        help='Learning rate of Actor Network (default: 1e-4)')
    parser.add_argument('--max_episode', type=float, default=1, metavar='N',
                        help='Max Episode (default: 200)')
    parser.add_argument('--noise_scale', type=float, default=0.1, metavar='G',
                        help='initial noise scale (default: 1.0)')
    parser.add_argument('--final_noise_scale', type=float, default=0.01, metavar='G',
                        help='final noise scale (default: 0.001)')

    parser.add_argument('--a_param', type=float, default=1.0, metavar='G',
                        help='a_param (default: 0.95)')
    parser.add_argument('--b_param', type=float, default=50.0, metavar='G',
                        help='b_param (default: 5.0~100.0)')
    
    args = parser.parse_args()
    
    for i in range(1):
        SEED = 1
        fixed_seed.fixed_seed_function(SEED)
        naf_environment = ENVIRONMENT(args, i)
        naf_environment.run()
    
    print("Learning Process Finished")
    
if __name__ == "__main__":
    main()