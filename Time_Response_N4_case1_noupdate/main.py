import torch 
import numpy as np
import argparse

from environment import ENVIRONMENT

def main():

    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    
    parser.add_argument('--hidden_layer_size', type=int, default=128, metavar='N',
                        help='Hidden Layer Size (default: 128)')
    parser.add_argument('--a_param', type=float, default=0.95, metavar='G',
                        help='dynamics a_parameter')
    parser.add_argument('--b_param', type=float, default=5.5, metavar='G',
                        help='dynamics b_parameter')
    
    args = parser.parse_args()
    
    for i in range(1):
        naf_environment = ENVIRONMENT(args, i)
        naf_environment.run()
    
    print("Learning Process Finished")
    
if __name__ == "__main__":
    np.random.seed(seed=0)
    main()