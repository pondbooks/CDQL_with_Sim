import torch 
import numpy as np
import argparse

from environment import ENVIRONMENT

def main():

    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--hidden_layer_size', type=int, default=128, metavar='N',
                        help='Hidden Layer Size (default: 128)')

    args = parser.parse_args()
    for i in range(1):
        naf_environment = ENVIRONMENT(args, i)
        naf_environment.run()

    
if __name__ == "__main__":
    main()